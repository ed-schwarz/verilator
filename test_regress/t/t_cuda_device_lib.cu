// DESCRIPTION: Verilator: Verilog Test CUDA device functions for t_cuda_device
//
// This file ONLY is placed under the Creative Commons Public Domain.
// SPDX-FileCopyrightText: 2026 Wilson Snyder
// SPDX-License-Identifier: CC0-1.0

#include <verilated.h>

__host__ __device__ void compute(QData a, QData b, IData* outp) {
    VlWide<4> lw;
    VlWide<4> rw;
    VlWide<4> ow;
    VL_CONCAT_WQQ(128, 64, 64, lw, a, b);
    VL_CONCAT_WQQ(128, 64, 64, rw, b, a);
    outp[0] = VL_COUNTONES_Q(a);
    outp[1] = VL_REDXOR_64(b);
    outp[2] = VL_CLOG2_Q(a);
    outp[3] = VL_DIVS_III(32, static_cast<IData>(a), static_cast<IData>(b) | 1);
    outp[4] = static_cast<IData>(VL_EXTENDS_QI(64, 7, 0x41) >> 32);
    VL_ADD_W(4, ow, lw, rw);
    outp[5] = ow[1] ^ ow[3];
    VL_MUL_W(4, ow, lw, rw);
    outp[6] = ow[2];
    outp[7] = ow[3];
    VL_SHIFTL_WWI(128, 128, 32, ow, lw, 37);
    outp[8] = ow[1];
    outp[9] = ow[2];
    outp[10] = VL_MOSTSETBITP1_W(4, lw);
    outp[11] = VL_EQ_W(4, lw, rw);
}
