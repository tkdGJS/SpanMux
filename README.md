# SpanMux 실험 매뉴얼

SpanMux는 단일 GPU에서 Prefill과 Decode를 분리하여 실행하고, Decode의
TBT 목표에 따라 SM을 배분합니다. 제안 기법은 SRPT, layer 경계 선점,
KV 회수 및 Context Ring을 함께 사용합니다.

이 가이드는 졸업논문에 사용한 **LoongServe Mixed 100요청, KV 2GiB,
최대 문맥 65,536토큰** 조건의 5개 비교군을 현재 코드에서 다시 실행하는
방법을 다룹니다. 논문 수치는 Baseline(real20), Baseline+SRPT·P/D+FCFS·
P/D+SRPT(real19), SpanMux(real22)의 보관 결과입니다. 이들도 같은
trace를 사용했지만 실행 당시의 소스 스냅샷과 바이너리는 달랐습니다.

## 1. 서버와 빌드 준비

Linux에서 CUDA와 호환되는 NVIDIA 드라이버, CUDA Toolkit(`nvcc`),
C++ 컴파일러, Python 가상 환경, Bash, Git, CMake와 Ninja가 필요합니다.
논문의 GPU는 Tesla T4(40 SM)입니다. GreenContext가 지원되는 드라이버를
사용하세요. 빌드 전에 `nvidia-smi`와 `nvcc --version`으로 환경을 확인합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate

# NVIDIA 환경과 호환되는 CUDA PyTorch 2.9.1이 설치되는지 확인하세요.
python -m pip install -r src/vllm-0.14-src/requirements/cuda.txt \
  -r src/vllm-0.14-src/requirements/build.txt
python -m pip install aiohttp
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'

(
  cd src/vllm-0.14-src
  CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8 VLLM_VERSION_OVERRIDE=0.14.0 \
    python -m pip install --no-build-isolation --no-deps -e .
)
```

`CUDA_HOME`은 서버에 설치한 Toolkit 경로로 바꿉니다. 기존에 동작한
가상 환경이 있다면 `thesis.env`의 `VENV_DIR`로 지정할 수 있습니다.
위 editable 설치본은 Baseline이 사용하는 vLLM과 수정된 CUDA 연산을
제공합니다. FlashAttention 소스는 CMake가 지정된 upstream 커밋에서
가져옵니다. 오프라인 빌드에서는 같은 커밋의 로컬 소스를 준비하고
`VLLM_FLASH_ATTN_SRC_DIR`로 경로를 지정하세요.

이어서 **같은 빌드에서 생성된 확장 모듈**을 SpanMux 패키지에 배치합니다.

```bash
find src/vllm-0.14-src/vllm -type f -name '*.so' -printf '%P\0' |
  rsync -a --from0 --files-from=- src/vllm-0.14-src/vllm/ spanmux/vllm/

VLLM_VENDOR_DIR="$PWD/spanmux" VLLM_PYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  bash tools/deploy_vendor_v8_direct_ops.sh

VLLM_VENDOR_DIR="$PWD/spanmux" \
  bash tools/green_context_smoke/build_vendor_v8_bridge.sh
```

`_C_cache_ops` 연산 검증에 실패하면 실험 전에 빌드와 배포 경로를
확인하세요. 모델 계정에 `meta-llama/Llama-3.2-1B-Instruct` 사용 권한이
있어야 합니다. 모델을 로컬에 준비했다면 설정 파일의 `MODEL_NAME`에 모델
디렉터리를 지정할 수 있습니다.

## 2. LoongServe trace 준비

기존 논문 trace가 있으면 `LOONGSERVE_TRACE_JSONL`에 그 파일을 지정하세요.
보관 trace의 SHA-256은
`2a0927f47d3d9e86547bf61f020d53236c04fdfb7f1e7064304e35663b7004ec`
입니다. 논문에서는 이 파일의 **처음 100개 요청을 파일 순서와 원래 도착
간격 그대로** 재생했습니다.

기존 trace가 없다면 ShareGPT, L-Eval, LV-Eval을 준비하고 토큰화하여
trace를 생성합니다. 아래 명령은 레포 루트에서 실행합니다.

```bash
prep_dir=exp_script
if [ ! -f "$prep_dir/prepare_loongserve_raw_datasets.sh" ]; then prep_dir=.; fi
LOONGSERVE_REPO_ROOT="$PWD" bash "$prep_dir/prepare_loongserve_raw_datasets.sh"

.venv/bin/python "$prep_dir/prepare_loongserve_mixed_workload.py" \
  --sharegpt-path data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --leval-path data/raw/loongserve/LEval/LEval-data \
  --lveval-path data/raw/loongserve/LVEval \
  --output-path data/loongserve_mixed_65k.jsonl \
  --tokenizer meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-workers 8 --tokenizer-batch-size 32 \
  --requests-per-dataset 1452 --request-rate 0.5 --seed 0 \
  --max-total-tokens 65536 --length-cap-policy filter_then_sample \
  --insufficient-policy error

sha256sum data/loongserve_mixed_65k.jsonl
```

총 길이 65,536을 넘는 요청을 제외한 뒤 데이터셋마다 1,452개를 뽑습니다.
출력 목표 길이도 trace에서 가져옵니다. 데이터나 tokenizer 버전에 따라
trace가 달라질 수 있으므로, 해시가 다르면 새 trace로 기록하세요.
`profiling/slo/request_slo_policy.json`에는 요청 길이별 기존 SLO 기준이
있습니다. 이는 workload 입력과 별개입니다.

## 3. 설정 파일로 실행

[`exp_script/thesis.env`](exp_script/thesis.env)에 공통 실험 조건을
모았습니다. 서버별 설정이 필요하면 복사해서 경로, 실험 이름 및 선택한
비교군을 수정하세요. 설정은 Bash가 읽는 로컬 파일입니다.

```bash
cp exp_script/thesis.env exp_script/thesis.local.env
bash exp_script/run_thesis.sh --config exp_script/thesis.local.env --dry-run
bash exp_script/run_thesis.sh --config exp_script/thesis.local.env
```

기본값을 그대로 쓰면 `bash exp_script/run_thesis.sh` 한 줄로 실행합니다.
`--dry-run`에서는 비교군별 `[PLAN]`과 서버 인수를 확인할 수 있습니다.
실제 실행은 비교군마다 서버를 시작하고 replay 후 종료합니다. 다시
돌릴 때는 `BASE_EXPERIMENT_NAME`을 새 값으로 바꿔 결과 위치를 구분하세요.

| 논문 표기 | `BENCHMARK_VARIANTS` 값 | 동작 |
|---|---|---|
| Baseline | `native_40sm_chunked` | Native 40SM mixed batch와 chunked Prefill |
| Baseline+SRPT | `baseline_plus` | Native mixed 실행에서 Prefill SRPT와 Decode FCFS |
| P/D+FCFS | `dynamic_hbm_serialized_fullsm` | P/D SM 분할, FCFS, KV 공간 부족 시 대기·직렬화 |
| P/D+SRPT | `dynamic_hbm_srpt` | P/D SM 분할, Prefill SRPT, KV 공간 부족 시 대기 |
| SpanMux | `full_vendor_v8` | P/D+SRPT, 배치의 layer 경계 선점, KV 회수 및 Context Ring |

단일 비교군만 점검하려면 복사한 설정에서 `BENCHMARK_VARIANTS`를
해당 ID 하나로, `LOONGSERVE_REQUEST_LIMIT`를 작은 값으로 바꾸세요.
논문과 비교할 때는 다섯 비교군에서 동일한 trace와 100요청으로 복원합니다.

| 설정 | 의미 |
|---|---|
| `LOONGSERVE_REQUEST_LIMIT=100` | trace 앞에서부터 제출할 요청 수. `requests` 모드는 첫 N건을 원래 도착 순서로 보내고 종료를 기다림 |
| `LOONGSERVE_ARRIVAL_TIME_SCALE=1` | 도착 간격 배수. 0.5는 두 배 빠른 도착, 2는 두 배 느린 도착 |
| `KV_CACHE_MEMORY_BYTES=2147483648` | KV 캐시 풀 2GiB. 모델·workspace를 포함한 전체 VRAM은 모니터링에서 별도 측정 |
| `VLLM_MAX_MODEL_LEN=65536` | 요청 하나의 전체 문맥 한도 |
| `VLLM_MAX_NUM_SEQS=100`, `VLLM_MAX_NUM_BATCHED_TOKENS=16384` | 한 iteration의 요청 수 및 토큰 상한. 실제 배치 폭은 현재 요청과 KV 용량으로 결정 |
| `PREFILL_CHUNK_TOKENS=2048`, `VLLM_LONG_PREFILL_TOKEN_THRESHOLD=2048` | guarded 2K 비교에서 요청별 Prefill 청크 상한을 맞춤. 16K 배치 토큰 예산과 별개 |
| `ISOLATION_GUARDED_2K_COMPARISON=1`, `ISOLATION_MATCHED_COMPARISON=1` | 비교군 공통 조건과 비교군별 보호·실행 기능을 런처에서 적용 |
| `BENCHMARK_CUDAGRAPH=1`, `FULL_AND_PIECEWISE` | CUDA Graph 설정. Native와 P/D의 실제 그래프 경로는 각 `[PLAN]`과 스냅샷에서 확인 |
| `PD_SM_PROFILES=0/40,30/10,...,40/0` | 후보 SM 배분. 항목은 Decode/Prefill 순서 |
| `PD_DYNAMIC_SM_POLICY=tbt_slo_feedback` | 프로파일링으로 정한 SLO와 관측 TBT에 따라 SM 배분을 조정 |
| `VLLM_PD_EVAL_*_INFLIGHT_DEPTH=1/1/2` | Prefill·Decode·전체 Future 예산. 배치 안의 요청 수와 별개 |
| `VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF=1` | 가능할 때 Prefill layer 경계에서 안전하게 SM 배분을 변경. 요청 교체와 KV 이동은 SpanMux variant 정책에서 별도로 활성화 |
| `FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY=scheduled_prefill_fit` | 다음 실행에 필요한 공간을 기준으로 Prefill KV 회수 |
| `FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION=0.98` | 회수 계획에서 Decode 성장 여유를 고려하는 용량 비율 |
| `VLLM_PD_EVAL_RING_COMPUTE_GROUP_TOKENS=2048` | Ring Attention에서 과거 KV를 계산하는 그룹 크기. 현재 Prefill 청크 크기와 별개 |
| `VLLM_PD_EVAL_*DIRECT_MULTICOPY`, `VLLM_PD_EVAL_VERIFY_DIRECT_OPS` | 직접 D2H/H2D 복사 사용 및 설치된 custom op 확인 |
| `WARMUP_COUNT=1`, `WARMUP_OUTPUT_TOKENS=8` | 측정 전에 워밍업 요청 1건, 출력 8토큰. 본 실험의 요청 집계에서 제외 |
| `REQUEST_TIMEOUT_SEC=28800`, `SERVER_READY_TIMEOUT_SEC=900` | 요청의 처리 제한과 서버 시작 제한을 각각 지정 |

## 4. 모니터링과 결과 확인

| 옵션 | 측정 내용 |
|---|---|
| `ENABLE_DMON=1` | 1초 간격 GPU 사용률, 프레임버퍼 메모리 사용량, 전력·온도·클록 등. `nvidia_smi_dmon.log`의 헤더 기준. 장치 수준 메모리이며 요청별 KV 점유량과는 별개 |
| `ENABLE_LIGHTWEIGHT_TIMELINE_TRACE=1` | iteration과 Decode timeline의 실행 단계·요청·SM profile 추적. 실제 출력 필드는 서버 variant와 지원 경로에 따라 다름 |
| `ITERATION_TRACE_SAMPLE_EVERY_N=100` | iteration 계측 샘플링 주기. 모든 요청의 클라이언트 TTFT·TBT 결과는 별도로 수집 |
| `ENABLE_SCHEDULER_TIMING_TRACE=1` | 지원 빌드에서 Scheduler 호출 시간 샘플을 `scheduler_metrics.jsonl`에 기록 |
| `ENABLE_PD_CUDA_EVENT_TRACE=1` | 지원 경로에서 CUDA event 기반 P/D 타이밍을 `pd_cuda_event_trace.jsonl`에 기록 |
| `ENABLE_FULL_ITERATION_TRACE=1`, `ENABLE_PD_EXECUTOR_DEBUG=1` | 상세 iteration·executor 진단. 계측 비용이 있어 기본 논문용 설정에서는 꺼둠 |

상세 계측을 켤 때는 런처의 조합 검사를 `--dry-run`으로 먼저 확인하세요.
엄격한 lightweight 추적은 상세 추적과 동시에 사용할 수 없는 경우가
있습니다. 성능 비교에는 비교군별로 같은 계측 조건을 적용하세요.

각 비교군은 `<실험이름>_workload_loongserve_<variant>` 접두사로 저장됩니다.

| 경로 | 확인할 값 |
|---|---|
| `*_results/client/run_metadata.json` | 실험 관측 시간, trace 해시, 요청 수와 종료 조건 |
| `*_results/client/request_results.csv` | 요청별 성공 여부, 입력·출력 토큰 수, TTFT, TBT, 요청 E2E, TTFT SLO 충족 여부 |
| `*_results/client/client_token_timeline.csv` | 스트리밍 토큰 간격, 토큰 수 가중치와 TBT SLO 충족 여부 |
| `*_results/client/workload_summary.csv` | 데이터셋별 성공·실패·미제출 수, 지연시간 분위수와 SLO 만족률 |
| `*_results/experiment_snapshot/` | 실제 적용된 설정, SLO 파일 및 실행 스크립트 스냅샷 |
| `*_artifacts/`, `*_logs/server.log` | GPU 모니터링 및 엔진 계측, 서버 오류와 시작·종료 로그 |

전체 실험 elapsed는 `observation_duration_s`를 사용합니다. 입력·출력
처리량은 완료 요청의 입력·출력 토큰 합계를 각각 이 시간으로 나눕니다.
TTFT 만족률은 요청 기준입니다. TBT 만족률은 첫 응답 chunk를 제외하고,
이후 chunk에서 추정한 토큰당 간격을 chunk의 토큰 수로 가중한 값입니다.
SLO goodput은 해당 기준을 만족한 요청 수 또는 토큰 수를 관측 시간으로
나눕니다. 완료 요청 수와 실패 원인을 처리량과 함께 확인하세요.
