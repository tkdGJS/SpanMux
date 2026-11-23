from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "run_sharegpt_mixed_context_motivation_4req_4GB.sh"


def test_runner_is_standalone_and_keeps_the_motivation_contract() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "run_real_workload_pd_vram_contention_5req_4GB.sh" not in script
    assert "prepare_mixed_context_sharegpt.py" in script
    assert "REQUEST_INPUT_TOKENS=(25000 50000 75000 100000 125000)" in script
    assert 'REQUEST_COUNT="${REQUEST_COUNT:-5}"' in script
    assert '--request-count "${REQUEST_COUNT}"' in script
    assert '--context-token-window "${MIXED_CONTEXT_TARGET_TOKENS}"' in script
    assert "MIXED_CONTEXT_TARGET_TOKENS=125000" in script
    assert 'SHAREGPT_UNDERLENGTH_POLICY="${SHAREGPT_UNDERLENGTH_POLICY:-pack}"' in script
    assert '--underlength-policy "${SHAREGPT_UNDERLENGTH_POLICY}"' in script
    assert "KV_CACHE_MEMORY_BYTES:-4294967296" in script
    assert (
        "muxwise_dynamic muxwise_dynamic_virtual_ring_optimized" in script
    )
    assert "VLLM_PD_EVAL_RELOAD_DIRECT_MULTICOPY:-1" in script
    assert "VLLM_PD_EVAL_RELOAD_STAGING_SCATTER:-0" in script
