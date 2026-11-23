// SPDX-License-Identifier: Apache-2.0
//
// Standalone CUDA Driver API smoke test for Green Contexts.
//
// This intentionally does not depend on vLLM or PyTorch. It verifies the first
// integration prerequisite:
//   device SM resource -> SM split -> green context -> green stream -> kernel.

#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifndef CUDA_VERSION
#error "CUDA_VERSION is not defined. Check CUDA Toolkit headers."
#endif

#if CUDA_VERSION < 12080
#error "CUDA 12.8 or newer headers are required for Green Context smoke test."
#endif

namespace {

const char *kFillPtx = R"ptx(
.version 7.5
.target sm_75
.address_size 64

.visible .entry fill_kernel(
    .param .u64 out_param,
    .param .u32 n_param,
    .param .u32 value_param
)
{
    .reg .pred %p;
    .reg .b32 %r<8>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [out_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r2, [value_param];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mad.lo.s32 %r6, %r4, %r5, %r3;

    setp.ge.u32 %p, %r6, %r1;
    @%p bra DONE;

    mul.wide.u32 %rd2, %r6, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r2;

DONE:
    ret;
}
)ptx";

std::string cuda_error(CUresult result) {
  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::string out = name ? name : "CUDA_ERROR_UNKNOWN";
  if (message != nullptr) {
    out += ": ";
    out += message;
  }
  return out;
}

bool check(CUresult result, const char *expr, const char *file, int line) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::cerr << "[FAIL] " << expr << " at " << file << ":" << line << " -> "
            << cuda_error(result) << "\n";
  return false;
}

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    if (!check((expr), #expr, __FILE__, __LINE__)) {                           \
      return 1;                                                                \
    }                                                                          \
  } while (false)

struct Options {
  int device = 0;
  unsigned int green_sms = 8;
  int priority = 0;
  unsigned int n = 4096;
};

void usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--device N] [--green-sms N] [--priority N] [--elements N]\n\n"
      << "Example:\n"
      << "  " << argv0 << " --device 0 --green-sms 8 --priority -1\n";
}

bool parse_uint(const char *text, unsigned int *value) {
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  *value = static_cast<unsigned int>(parsed);
  return true;
}

bool parse_int(const char *text, int *value) {
  char *end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool parse_args(int argc, char **argv, Options *options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--device") {
      const char *value = need_value("--device");
      if (value == nullptr || !parse_int(value, &options->device)) {
        return false;
      }
    } else if (arg == "--green-sms") {
      const char *value = need_value("--green-sms");
      if (value == nullptr || !parse_uint(value, &options->green_sms)) {
        return false;
      }
    } else if (arg == "--priority") {
      const char *value = need_value("--priority");
      if (value == nullptr || !parse_int(value, &options->priority)) {
        return false;
      }
    } else if (arg == "--elements") {
      const char *value = need_value("--elements");
      if (value == nullptr || !parse_uint(value, &options->n)) {
        return false;
      }
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse_args(argc, argv, &options)) {
    usage(argv[0]);
    return 2;
  }

  CHECK_CUDA(cuInit(0));

  int device_count = 0;
  CHECK_CUDA(cuDeviceGetCount(&device_count));
  if (options.device < 0 || options.device >= device_count) {
    std::cerr << "[FAIL] invalid device index " << options.device
              << ", device_count=" << device_count << "\n";
    return 1;
  }

  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, options.device));

  char device_name[256] = {};
  CHECK_CUDA(cuDeviceGetName(device_name, sizeof(device_name), device));

  int sm_count = 0;
  CHECK_CUDA(cuDeviceGetAttribute(&sm_count,
                                  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                  device));
  int cc_major = 0;
  int cc_minor = 0;
  CHECK_CUDA(cuDeviceGetAttribute(&cc_major,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                  device));
  CHECK_CUDA(cuDeviceGetAttribute(&cc_minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                  device));

  std::cout << "[INFO] device=" << options.device << " name=" << device_name
            << " cc=" << cc_major << "." << cc_minor
            << " sms=" << sm_count << "\n";
  std::cout << "[INFO] requested_green_sms=" << options.green_sms
            << " priority=" << options.priority << "\n";

  if (options.green_sms < 2 ||
      options.green_sms > static_cast<unsigned int>(sm_count)) {
    std::cerr << "[FAIL] --green-sms must be in [2, " << sm_count << "]\n";
    return 1;
  }

  CUcontext primary = nullptr;
  CHECK_CUDA(cuDevicePrimaryCtxRetain(&primary, device));
  CHECK_CUDA(cuCtxSetCurrent(primary));

  CUdevResource sm_resource = {};
  CHECK_CUDA(
      cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));

  CUdevResource green_sm_resource = {};
  CUdevResource remaining_sm_resource = {};
  unsigned int num_groups = 1;
  CHECK_CUDA(cuDevSmResourceSplitByCount(
      &green_sm_resource, &num_groups, &sm_resource, &remaining_sm_resource,
      0, options.green_sms));
  if (num_groups == 0) {
    std::cerr << "[FAIL] cuDevSmResourceSplitByCount returned zero groups\n";
    return 1;
  }
  std::cout << "[INFO] split_groups_created=" << num_groups << "\n";

  CUdevResourceDesc resource_desc = {};
  CHECK_CUDA(cuDevResourceGenerateDesc(&resource_desc, &green_sm_resource, 1));

  CUgreenCtx green_ctx = nullptr;
  CHECK_CUDA(cuGreenCtxCreate(&green_ctx, resource_desc, device,
                              CU_GREEN_CTX_DEFAULT_STREAM));

  CUcontext green_as_context = nullptr;
  CHECK_CUDA(cuCtxFromGreenCtx(&green_as_context, green_ctx));
  CHECK_CUDA(cuCtxSetCurrent(green_as_context));

  CUstream green_stream = nullptr;
  CHECK_CUDA(cuGreenCtxStreamCreate(&green_stream, green_ctx,
                                    CU_STREAM_NON_BLOCKING, options.priority));

  CUgreenCtx stream_green_ctx = nullptr;
  CHECK_CUDA(cuStreamGetGreenCtx(green_stream, &stream_green_ctx));
  if (stream_green_ctx != green_ctx) {
    std::cerr << "[FAIL] cuStreamGetGreenCtx returned a different green ctx\n";
    return 1;
  }

  CUmodule module = nullptr;
  CUfunction kernel = nullptr;
  CHECK_CUDA(cuModuleLoadDataEx(&module, kFillPtx, 0, nullptr, nullptr));
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, "fill_kernel"));

  CUdeviceptr device_buffer = 0;
  const size_t bytes = static_cast<size_t>(options.n) * sizeof(uint32_t);
  CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));

  uint32_t expected = 0x5a5a1234u;
  void *kernel_args[] = {&device_buffer, &options.n, &expected};
  const unsigned int threads = 256;
  const unsigned int blocks = (options.n + threads - 1) / threads;
  CHECK_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0,
                            green_stream, kernel_args, nullptr));

  std::vector<uint32_t> host(options.n, 0);
  CHECK_CUDA(cuMemcpyDtoHAsync(host.data(), device_buffer, bytes, green_stream));
  CHECK_CUDA(cuStreamSynchronize(green_stream));

  for (unsigned int i = 0; i < options.n; ++i) {
    if (host[i] != expected) {
      std::cerr << "[FAIL] output mismatch at " << i << ": got=0x" << std::hex
                << host[i] << " expected=0x" << expected << std::dec << "\n";
      return 1;
    }
  }

  CHECK_CUDA(cuMemFree(device_buffer));
  CHECK_CUDA(cuModuleUnload(module));
  CHECK_CUDA(cuStreamDestroy(green_stream));
  CHECK_CUDA(cuGreenCtxDestroy(green_ctx));
  CHECK_CUDA(cuCtxSetCurrent(primary));
  CHECK_CUDA(cuDevicePrimaryCtxRelease(device));

  std::cout << "[OK] Green Context stream created and kernel validated\n";
  return 0;
}
