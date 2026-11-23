// SPDX-License-Identifier: Apache-2.0
//
// C ABI bridge for vendor_v7 Green Context experiments.
//
// This library owns CUDA Driver API Green Context objects and exposes only raw
// CUstream pointers to Python. It is intentionally small and default-off.

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifndef CUDA_VERSION
#error "CUDA_VERSION is not defined. Check CUDA Toolkit headers."
#endif

#if CUDA_VERSION < 12080
#error "CUDA 12.8 or newer headers are required for Green Context bridge."
#endif

namespace {

constexpr int kPhaseDecode = 0;
constexpr int kPhasePrefill = 1;

struct VllmGreenPDHandle {
  CUdevice device = 0;
  CUcontext previous_context = nullptr;
  CUcontext primary_context = nullptr;
  CUgreenCtx decode_green_context = nullptr;
  CUcontext decode_context = nullptr;
  CUstream decode_stream = nullptr;
  CUgreenCtx prefill_green_context = nullptr;
  CUcontext prefill_context = nullptr;
  CUstream prefill_stream = nullptr;
  unsigned int decode_sms = 0;
  unsigned int prefill_sms = 0;
};

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

void set_error(char *error, size_t error_size, const std::string &message) {
  if (error == nullptr || error_size == 0) {
    return;
  }
  std::snprintf(error, error_size, "%s", message.c_str());
}

bool check_cuda(CUresult result,
                const char *expr,
                char *error,
                size_t error_size) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::ostringstream oss;
  oss << expr << " failed: " << cuda_error(result);
  set_error(error, error_size, oss.str());
  return false;
}

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    if (!check_cuda((expr), #expr, error, error_size)) {                       \
      return false;                                                            \
    }                                                                          \
  } while (false)

bool create_stream_from_resources(CUdevice device,
                                  const CUdevResource *resources,
                                  unsigned int resource_count,
                                  int priority,
                                  CUgreenCtx *green_context,
                                  CUcontext *context,
                                  CUstream *stream,
                                  char *error,
                                  size_t error_size) {
  if (resources == nullptr || resource_count == 0) {
    set_error(error, error_size, "Green Context resource list is empty");
    return false;
  }
  CUdevResourceDesc resource_desc = {};
  std::vector<CUdevResource> resource_copies(resources,
                                             resources + resource_count);
  CHECK_CUDA(cuDevResourceGenerateDesc(&resource_desc, resource_copies.data(),
                                       resource_count));
  CHECK_CUDA(cuGreenCtxCreate(green_context, resource_desc, device,
                              CU_GREEN_CTX_DEFAULT_STREAM));
  CHECK_CUDA(cuCtxFromGreenCtx(context, *green_context));
  CHECK_CUDA(cuCtxSetCurrent(*context));
  CHECK_CUDA(cuGreenCtxStreamCreate(stream, *green_context,
                                    CU_STREAM_NON_BLOCKING, priority));
  return true;
}

bool split_sm_resources(const CUdevResource &input,
                        unsigned int group_sm_count,
                        unsigned int requested_groups,
                        std::vector<CUdevResource> *selected,
                        CUdevResource *remaining,
                        char *error,
                        size_t error_size) {
  if (selected == nullptr || remaining == nullptr) {
    set_error(error, error_size, "split_sm_resources output pointer is null");
    return false;
  }
  if (group_sm_count == 0 || requested_groups == 0) {
    set_error(error, error_size,
              "split_sm_resources requires positive group size and count");
    return false;
  }
  selected->assign(requested_groups, CUdevResource{});
  unsigned int num_groups = requested_groups;
  CHECK_CUDA(cuDevSmResourceSplitByCount(selected->data(), &num_groups, &input,
                                         remaining, 0, group_sm_count));
  if (num_groups < requested_groups) {
    std::ostringstream oss;
    oss << "cuDevSmResourceSplitByCount returned " << num_groups
        << " groups, requested " << requested_groups
        << " groups of " << group_sm_count << " SMs";
    set_error(error, error_size, oss.str());
    return false;
  }
  return true;
}

bool create_single_sm_stream(CUdevice device,
                             const CUdevResource &available_sm_resource,
                             unsigned int sms,
                             int priority,
                             CUgreenCtx *green_context,
                             CUcontext *context,
                             CUstream *stream,
                             char *error,
                             size_t error_size) {
  CUdevResource remaining_sm_resource = {};
  std::vector<CUdevResource> selected_sm_resources;
  if (!split_sm_resources(available_sm_resource, sms, 1,
                          &selected_sm_resources, &remaining_sm_resource,
                          error, error_size)) {
    return false;
  }
  return create_stream_from_resources(device, selected_sm_resources.data(), 1,
                                      priority, green_context, context, stream,
                                      error, error_size);
}

void destroy_noexcept(VllmGreenPDHandle *handle) {
  if (handle == nullptr) {
    return;
  }

  if (handle->decode_context != nullptr) {
    cuCtxSetCurrent(handle->decode_context);
  }
  if (handle->decode_stream != nullptr) {
    cuStreamDestroy(handle->decode_stream);
  }
  if (handle->decode_green_context != nullptr) {
    cuGreenCtxDestroy(handle->decode_green_context);
  }

  if (handle->prefill_context != nullptr) {
    cuCtxSetCurrent(handle->prefill_context);
  }
  if (handle->prefill_stream != nullptr) {
    cuStreamDestroy(handle->prefill_stream);
  }
  if (handle->prefill_green_context != nullptr) {
    cuGreenCtxDestroy(handle->prefill_green_context);
  }

  cuCtxSetCurrent(handle->previous_context);
  if (handle->primary_context != nullptr) {
    cuDevicePrimaryCtxRelease(handle->device);
  }
  delete handle;
}

}  // namespace

extern "C" {

int vllm_green_pd_create(int device_index,
                         unsigned int decode_sms,
                         unsigned int prefill_sms,
                         int decode_priority,
                         int prefill_priority,
                         VllmGreenPDHandle **out,
                         char *error,
                         size_t error_size) {
  if (out == nullptr) {
    set_error(error, error_size, "out handle pointer is null");
    return 1;
  }
  *out = nullptr;
  if (decode_sms == 0 && prefill_sms == 0) {
    set_error(error, error_size,
              "at least one of decode_sms or prefill_sms must be positive");
    return 1;
  }

  if (!check_cuda(cuInit(0), "cuInit", error, error_size)) {
    return 1;
  }

  int device_count = 0;
  if (!check_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount", error,
                  error_size)) {
    return 1;
  }
  if (device_index < 0 || device_index >= device_count) {
    std::ostringstream oss;
    oss << "invalid device index " << device_index
        << ", device_count=" << device_count;
    set_error(error, error_size, oss.str());
    return 1;
  }

  VllmGreenPDHandle *handle = new VllmGreenPDHandle();
  if (!check_cuda(cuCtxGetCurrent(&handle->previous_context),
                  "cuCtxGetCurrent", error, error_size) ||
      !check_cuda(cuDeviceGet(&handle->device, device_index), "cuDeviceGet",
                  error, error_size)) {
    destroy_noexcept(handle);
    return 1;
  }

  int sm_count = 0;
  if (!check_cuda(cuDeviceGetAttribute(
                      &sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                      handle->device),
                  "cuDeviceGetAttribute(SM count)", error, error_size)) {
    destroy_noexcept(handle);
    return 1;
  }
  if (decode_sms + prefill_sms > static_cast<unsigned int>(sm_count)) {
    std::ostringstream oss;
    oss << "requested Green Context SMs exceed device SMs: decode_sms="
        << decode_sms << " prefill_sms=" << prefill_sms
        << " device_sms=" << sm_count;
    set_error(error, error_size, oss.str());
    destroy_noexcept(handle);
    return 1;
  }

  if (!check_cuda(cuDevicePrimaryCtxRetain(&handle->primary_context,
                                           handle->device),
                  "cuDevicePrimaryCtxRetain", error, error_size) ||
      !check_cuda(cuCtxSetCurrent(handle->primary_context),
                  "cuCtxSetCurrent(primary)", error, error_size)) {
    destroy_noexcept(handle);
    return 1;
  }

  CUdevResource available_sm_resource = {};
  if (!check_cuda(cuDeviceGetDevResource(handle->device, &available_sm_resource,
                                         CU_DEV_RESOURCE_TYPE_SM),
                  "cuDeviceGetDevResource", error, error_size)) {
    destroy_noexcept(handle);
    return 1;
  }

  if (decode_sms > 0 && prefill_sms > 0) {
    const unsigned int group_sms = std::gcd(decode_sms, prefill_sms);
    if (group_sms == 0) {
      set_error(error, error_size, "invalid zero SM group size");
      destroy_noexcept(handle);
      return 1;
    }
    const unsigned int decode_groups = decode_sms / group_sms;
    const unsigned int prefill_groups = prefill_sms / group_sms;
    const unsigned int total_groups = decode_groups + prefill_groups;

    CUdevResource remaining_sm_resource = {};
    std::vector<CUdevResource> split_groups;
    if (!split_sm_resources(available_sm_resource, group_sms, total_groups,
                            &split_groups, &remaining_sm_resource, error,
                            error_size) ||
        !create_stream_from_resources(handle->device, split_groups.data(),
                                      decode_groups, decode_priority,
                                      &handle->decode_green_context,
                                      &handle->decode_context,
                                      &handle->decode_stream, error,
                                      error_size) ||
        !create_stream_from_resources(
            handle->device, split_groups.data() + decode_groups,
            prefill_groups, prefill_priority, &handle->prefill_green_context,
            &handle->prefill_context, &handle->prefill_stream, error,
            error_size)) {
      destroy_noexcept(handle);
      return 1;
    }
    handle->decode_sms = decode_sms;
    handle->prefill_sms = prefill_sms;
  } else if (decode_sms > 0) {
    if (!create_single_sm_stream(handle->device, available_sm_resource,
                                 decode_sms, decode_priority,
                                 &handle->decode_green_context,
                                 &handle->decode_context,
                                 &handle->decode_stream, error, error_size)) {
      destroy_noexcept(handle);
      return 1;
    }
    handle->decode_sms = decode_sms;
  } else if (prefill_sms > 0) {
    if (!create_single_sm_stream(handle->device, available_sm_resource,
                                 prefill_sms, prefill_priority,
                                 &handle->prefill_green_context,
                                 &handle->prefill_context,
                                 &handle->prefill_stream, error, error_size)) {
      destroy_noexcept(handle);
      return 1;
    }
    handle->prefill_sms = prefill_sms;
  }

  if (!check_cuda(cuCtxSetCurrent(handle->previous_context),
                  "cuCtxSetCurrent(previous)", error, error_size)) {
    destroy_noexcept(handle);
    return 1;
  }
  *out = handle;
  return 0;
}

int vllm_green_pd_destroy(VllmGreenPDHandle *handle) {
  destroy_noexcept(handle);
  return 0;
}

int vllm_green_pd_synchronize(VllmGreenPDHandle *handle) {
  if (handle == nullptr) {
    return 1;
  }
  CUcontext previous_context = nullptr;
  if (cuCtxGetCurrent(&previous_context) != CUDA_SUCCESS) {
    return 1;
  }

  int status = 0;
  if (handle->decode_stream != nullptr &&
      (cuCtxSetCurrent(handle->decode_context) != CUDA_SUCCESS ||
       cuStreamSynchronize(handle->decode_stream) != CUDA_SUCCESS)) {
    status = 1;
  }
  if (handle->prefill_stream != nullptr &&
      (cuCtxSetCurrent(handle->prefill_context) != CUDA_SUCCESS ||
       cuStreamSynchronize(handle->prefill_stream) != CUDA_SUCCESS)) {
    status = 1;
  }
  if (cuCtxSetCurrent(previous_context) != CUDA_SUCCESS) {
    status = 1;
  }
  return status;
}

unsigned long long vllm_green_pd_stream(VllmGreenPDHandle *handle, int phase) {
  if (handle == nullptr) {
    return 0;
  }
  if (phase == kPhaseDecode) {
    return reinterpret_cast<unsigned long long>(handle->decode_stream);
  }
  if (phase == kPhasePrefill) {
    return reinterpret_cast<unsigned long long>(handle->prefill_stream);
  }
  return 0;
}

int vllm_green_pd_sm_count(VllmGreenPDHandle *handle, int phase) {
  if (handle == nullptr) {
    return -1;
  }
  CUgreenCtx green_context = nullptr;
  if (phase == kPhaseDecode) {
    green_context = handle->decode_green_context;
  } else if (phase == kPhasePrefill) {
    green_context = handle->prefill_green_context;
  } else {
    return -1;
  }
  if (green_context == nullptr) {
    return 0;
  }

  CUdevResource resource = {};
  if (cuGreenCtxGetDevResource(green_context, &resource,
                               CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS ||
      resource.type != CU_DEV_RESOURCE_TYPE_SM) {
    return -1;
  }
  return static_cast<int>(resource.sm.smCount);
}

}  // extern "C"
