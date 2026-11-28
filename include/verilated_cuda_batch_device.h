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
/// \brief GPU runtime: device side
///
//=============================================================================

#ifndef VERILATOR_VERILATED_CUDA_BATCH_DEVICE_H_
#define VERILATOR_VERILATED_CUDA_BATCH_DEVICE_H_

#include "verilated.h"
#include "verilated_cuda_batch.h"

#include <cuda_runtime.h>

struct VlCudaBatchView final {
    uint8_t* m_base;  // Base of the state of this copy (or of all copies, by layout)
    uint32_t m_n;  // Number of copies
    uint32_t m_i;  // Index of this copy
};

template <typename T>
__device__ __forceinline__ T& vlCudaBatchSoa(const VlCudaBatchView& view, size_t offset) {
#if defined(VL_CUDA_BATCH_LAYOUT_AOS)
    return *reinterpret_cast<T*>(view.m_base + offset);
#elif defined(VL_CUDA_BATCH_LAYOUT_AOSOA)
    return *reinterpret_cast<T*>(view.m_base + VL_CUDA_BATCH_LAYOUT_GROUP * offset
                                 + (view.m_i % VL_CUDA_BATCH_LAYOUT_GROUP) * sizeof(T));
#else
    return *reinterpret_cast<T*>(view.m_base + static_cast<size_t>(view.m_n) * offset
                                 + static_cast<size_t>(view.m_i) * sizeof(T));
#endif
}

extern __device__ unsigned int vlCudaBatchDeferPending;
extern __device__ uint64_t vlCudaBatchNow;
extern __device__ VlCudaBatchAssertCtl vlCudaBatchAssertCtlState;

__host__ __device__ constexpr uint32_t
vlCudaBatchAssertOnMask(VerilatedAssertType_t types, VerilatedAssertDirectiveType_t directives) {
    uint32_t mask = 0;
    for (int i = 0; i < 8 * static_cast<int>(sizeof(VerilatedAssertType_t)); ++i) {
        if ((types >> i) & 1) mask |= static_cast<uint32_t>(directives) << (i * 3);
    }
    return mask;
}

template <VerilatedAssertType_t T_Type, VerilatedAssertDirectiveType_t T_Directive>
__device__ __forceinline__ uint32_t vlCudaBatchAssertCtlGet(VerilatedAssertCtlQuery query) {
    constexpr uint32_t mask = vlCudaBatchAssertOnMask(T_Type, T_Directive);
    if (!mask) return 0;
    switch (query) {
    case VerilatedAssertCtlQuery::ASSERT_CTL_ON:
        return (vlCudaBatchAssertCtlState.m_on & mask) != 0;
    case VerilatedAssertCtlQuery::ASSERT_CTL_KILL:
        return vlCudaBatchAssertCtlState.m_kill[__ffs(mask) - 1];
    case VerilatedAssertCtlQuery::ASSERT_CTL_PASS_ON_VACUOUS:
        return (vlCudaBatchAssertCtlState.m_passVacuous & mask) != 0;
    case VerilatedAssertCtlQuery::ASSERT_CTL_PASS_ON_NONVACUOUS:
        return (vlCudaBatchAssertCtlState.m_passNonvacuous & mask) != 0;
    case VerilatedAssertCtlQuery::ASSERT_CTL_FAIL_ON:
        return (vlCudaBatchAssertCtlState.m_failOn & mask) != 0;
    default: return 0;
    }
}

struct VlCudaBatchDpiCtx final {
    uint32_t m_copy;  // Index of the calling copy
    uint32_t m_n;  // Number of copies
    VlCudaBatchDeferBuf* m_deferp;  // Output buffer of the calling copy
};

__device__ inline uint8_t* vlCudaBatchDeferBegin(VlCudaBatchDeferBuf& buf, uint32_t id,
                                                 uint32_t bytes) {
    const uint32_t need = 16 + ((bytes + 3U) & ~3U);
    atomicAdd(&vlCudaBatchDeferPending, 1U);
    if (buf.m_used + need > VlCudaBatchDeferBuf::SIZE) {
        buf.m_overflow = 1;
        return nullptr;
    }
    uint8_t* const recp = buf.m_data + buf.m_used;
    buf.m_used += need;
    const uint64_t now = vlCudaBatchNow;
    memcpy(recp, &id, 4);
    memcpy(recp + 4, &bytes, 4);
    memcpy(recp + 8, &now, 8);
    return recp + 16;
}

__device__ inline void vlCudaBatchDpiWrite(const VlCudaBatchDpiCtx& ctx, const char* datap,
                                           uint32_t len) {
    uint8_t* const p = vlCudaBatchDeferBegin(*ctx.m_deferp, 0xffffffffU, len);
    if (p) memcpy(p, datap, len);
}

#endif
