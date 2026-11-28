// DESCRIPTION: Verilator: Verilog Test CUDA main for t_cuda_device
//
// This file ONLY is placed under the Creative Commons Public Domain.
// SPDX-FileCopyrightText: 2026 Wilson Snyder
// SPDX-License-Identifier: CC0-1.0

#include <verilated.h>

#include <cstdio>
#include <cuda_runtime_api.h>
#include <memory>

#include VM_PREFIX_INCLUDE

constexpr int RESULTS = 12;

__host__ __device__ void compute(QData a, QData b, IData* outp);

__global__ void kernel(QData a, QData b, IData* outp) { compute(a, b, outp); }

static bool checkDevice() {
    const QData a = 0x8765432112345678ULL;
    const QData b = 0x0fedcba987654321ULL;
    IData host[RESULTS];
    compute(a, b, host);
    IData* devp = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&devp), sizeof(host)) != cudaSuccess) return false;
    kernel<<<1, 1>>>(a, b, devp);
    IData dev[RESULTS];
    const bool ok = cudaMemcpy(dev, devp, sizeof(dev), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(devp);
    if (!ok) return false;
    bool same = true;
    for (int i = 0; i < RESULTS; ++i) {
        if (dev[i] != host[i]) {
            std::printf("%%Error: result %d: device=%08x host=%08x\n", i, dev[i], host[i]);
            same = false;
        }
    }
    return same;
}

int main(int argc, char** argv) {
    const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
    contextp->commandArgs(argc, argv);
    const std::unique_ptr<VM_PREFIX> topp{new VM_PREFIX{contextp.get()}};

    if (!checkDevice()) {
        std::printf("%%Error: device results differ\n");
        return 1;
    }

    topp->clk = 0;
    while (!contextp->gotFinish() && contextp->time() < 1000) {
        topp->eval();
        contextp->timeInc(1);
        topp->clk = !topp->clk;
    }
    topp->final();
    return 0;
}
