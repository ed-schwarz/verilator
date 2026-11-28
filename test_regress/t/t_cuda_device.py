#!/usr/bin/env python3
# DESCRIPTION: Verilator: Verilog Test driver/expect definition
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of either the GNU Lesser General Public License Version 3
# or the Perl Artistic License Version 2.0.
# SPDX-FileCopyrightText: 2026 Wilson Snyder
# SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0

import vltest_bootstrap

import os
import shutil

test.scenarios('vlt')

if not test.have_gpu:
    test.skip("No CUDA compiler or GPU")

cuda_lib = os.path.join(os.path.dirname(os.path.dirname(shutil.which("nvcc"))), "lib64")

test.compile(make_main=False,
             verilator_flags2=[
                 "--exe", "--compiler nvcc", "t/t_cuda_device.cu", "-LDFLAGS",
                 "'-L" + cuda_lib + " -lcudart'"
             ])

test.execute()

test.passes()
