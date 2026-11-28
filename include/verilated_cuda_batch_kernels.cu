// -*- mode: C++; c-file-style: "cc-mode" -*-
//=============================================================================
//
// Code available from: https://verilator.org
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of either the GNU Lesser General Public License Version 3
// or the Perl Artistic License Version 2.0.
// SPDX-FileCopyrightText: 2026-2026 Wilson Snyder
// SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0
//
//=============================================================================
///
/// \file
/// \brief GPU runtime: device kernels
///
//=============================================================================

#include "verilated_cuda_batch.h"
#include "verilated_cuda_batch_device.h"

#include <cuda_runtime.h>

static __global__ void vlCudaBatchProbeKernel(int* outp) { *outp = 1; }

extern "C" int vlCudaBatchProbe() {
    int* devp = nullptr;
    if (cudaMalloc(&devp, sizeof(int)) != cudaSuccess) return -1;
    vlCudaBatchProbeKernel<<<1, 1>>>(devp);
    int host = 0;
    const cudaError_t err = cudaMemcpy(&host, devp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devp);
    if (err != cudaSuccess) return -2;
    return host == 1 ? 0 : -3;
}

static __global__ void vlCudaBatchScatterKernel(const uint8_t* aosp, uint8_t* soap, int n,
                                                int copy, const VlCudaBatchCopy* tablep, int count,
                                                size_t stride, int mode) {
    const int member = blockIdx.x;
    if (member >= count) return;
    const VlCudaBatchCopy entry = tablep[member];
    const uint8_t* const fromp = aosp + entry.m_state;
    const int first = copy < 0 ? 0 : copy;
    const int last = copy < 0 ? n : copy + 1;
    for (int i = first + threadIdx.x; i < last; i += blockDim.x) {
        const size_t copy = static_cast<size_t>(i);
        uint8_t* const top
            = mode == 1 ? soap + copy * stride + entry.m_state
              : mode == 2
                  ? soap
                        + (copy / VL_CUDA_BATCH_LAYOUT_GROUP) * VL_CUDA_BATCH_LAYOUT_GROUP * stride
                        + VL_CUDA_BATCH_LAYOUT_GROUP * entry.m_state
                        + (copy % VL_CUDA_BATCH_LAYOUT_GROUP) * entry.m_size
                  : soap + static_cast<size_t>(n) * entry.m_state + copy * entry.m_size;
        for (size_t b = 0; b < entry.m_size; ++b) top[b] = fromp[b];
    }
}

extern "C" int vlCudaBatchScatter(const void* aosp, void* soap, int n, int copy,
                                  const void* tablep, int count, size_t stride, int mode) {
    if (count <= 0) return 0;
    vlCudaBatchScatterKernel<<<count, 128>>>(
        static_cast<const uint8_t*>(aosp), static_cast<uint8_t*>(soap), n, copy,
        static_cast<const VlCudaBatchCopy*>(tablep), count, stride, mode);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

__device__ unsigned int vlCudaBatchDeferPending = 0;
__device__ uint64_t vlCudaBatchNow = 0;

extern "C" int vlCudaBatchNowWrite(uint64_t now) {
    return cudaMemcpyToSymbol(vlCudaBatchNow, &now, sizeof(now)) == cudaSuccess ? 0 : -1;
}

extern "C" int vlCudaBatchNowRead(uint64_t* nowp) {
    return cudaMemcpyFromSymbol(nowp, vlCudaBatchNow, sizeof(*nowp)) == cudaSuccess ? 0 : -1;
}
__device__ VlCudaBatchAssertCtl vlCudaBatchAssertCtlState;

extern "C" int vlCudaBatchAssertCtlUpload(const void* ctlp) {
    return cudaMemcpyToSymbol(vlCudaBatchAssertCtlState, ctlp, sizeof(VlCudaBatchAssertCtl))
                   == cudaSuccess
               ? 0
               : -1;
}

static __global__ void vlCudaBatchCheckKernel(const uint8_t* soap, int n,
                                              const VlCudaBatchCopy* tablep, int count,
                                              uint8_t* leaderp, unsigned int* statusp, int check,
                                              int leader, const uint8_t* activep) {
    const int member = blockIdx.x;
    if (member >= count) return;
    const VlCudaBatchCopy entry = tablep[member];
    const uint8_t* const basep = soap + static_cast<size_t>(n) * entry.m_state;
    const uint8_t* const firstp = basep + static_cast<size_t>(leader) * entry.m_size;
    if (threadIdx.x == 0) {
        for (size_t b = 0; b < entry.m_size; ++b) leaderp[entry.m_state + b] = firstp[b];
    }
    for (int i = threadIdx.x; check && i < n; i += blockDim.x) {
        if (i == leader || !activep[i]) continue;
        const uint8_t* const cp = basep + static_cast<size_t>(i) * entry.m_size;
        for (size_t b = 0; b < entry.m_size; ++b) {
            if (cp[b] != firstp[b]) {
                atomicOr(&statusp[1], 1U);
                atomicMin(&statusp[2], static_cast<unsigned int>(i));
                break;
            }
        }
    }
    if (member == 0 && threadIdx.x == 0) statusp[0] = vlCudaBatchDeferPending;
}

extern "C" int vlCudaBatchBroadcastAsync(const void* aosp, void* soap, int n, const void* tablep,
                                         int count) {
    if (count <= 0) return 0;
    vlCudaBatchScatterKernel<<<count, 128>>>(
        static_cast<const uint8_t*>(aosp), static_cast<uint8_t*>(soap), n, -1,
        static_cast<const VlCudaBatchCopy*>(tablep), count, 0, 0);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static __global__ void vlCudaBatchStatusInit(unsigned int* statusp) {
    statusp[0] = 0;
    statusp[1] = 0;
    statusp[2] = 0xffffffffU;
}

extern "C" int vlCudaBatchCheckAsync(const void* soap, int n, const void* tablep, int count,
                                     void* leaderp, void* statusp, int check, int leader,
                                     const void* activep) {
    vlCudaBatchStatusInit<<<1, 1>>>(static_cast<unsigned int*>(statusp));
    if (count > 0) {
        vlCudaBatchCheckKernel<<<count, 128>>>(
            static_cast<const uint8_t*>(soap), n, static_cast<const VlCudaBatchCopy*>(tablep),
            count, static_cast<uint8_t*>(leaderp), static_cast<unsigned int*>(statusp), check,
            leader, static_cast<const uint8_t*>(activep));
    } else {
        unsigned int pending = 0;
        if (cudaMemcpyFromSymbol(&pending, vlCudaBatchDeferPending, sizeof(pending))
            != cudaSuccess)
            return -1;
        if (cudaMemcpy(statusp, &pending, sizeof(pending), cudaMemcpyHostToDevice) != cudaSuccess)
            return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int vlCudaBatchDeferPendingReset() {
    const unsigned int zero = 0;
    return cudaMemcpyToSymbol(vlCudaBatchDeferPending, &zero, sizeof(zero)) == cudaSuccess ? 0
                                                                                           : -1;
}
