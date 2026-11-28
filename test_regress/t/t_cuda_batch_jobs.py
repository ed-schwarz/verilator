#!/usr/bin/env python3
# DESCRIPTION: Verilator: Verilog Test driver/expect definition
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of either the GNU Lesser General Public License Version 3
# or the Perl Artistic License Version 2.0.
# SPDX-FileCopyrightText: 2026 Wilson Snyder
# SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0

import vltest_bootstrap

test.scenarios('vlt')

if not test.have_gpu:
    test.skip("No CUDA GPU available")

test.compile(verilator_flags2=[
    "--binary", "--cuda-batch", "--cuda-batch-dpi", "job_seed", test.t_dir + "/" + test.name +
    ".cu"
])

jobs = test.obj_dir + "/jobs.txt"
test.write_wholefile(
    jobs, "first +seed=3\n"
    "second +seed=7 +limit=45\n"
    "stopped +seed=5\n"
    "third +seed=11\n")

test.execute(run_env="VERILATOR_CUDA_BATCH_JOBS=" + jobs + " VERILATOR_CUDA_BATCH_LOGDIR=" +
             test.obj_dir + " VERILATOR_CUDA_BATCH_SLOTS=2",
             check_finished=False)

results = test.obj_dir + "/results.log"
with open(results, "w", encoding="utf8") as out:
    for name in ["cuda_batch_summary", "first", "second", "stopped", "third"]:
        out.write("== " + name + "\n")
        with open(test.obj_dir + "/" + name +
                  (".txt" if name.startswith("cuda_batch") else ".log"),
                  encoding="utf8") as fh:
            out.write(fh.read())
test.files_identical(results, test.golden_filename)

test.execute(run_env="VERILATOR_CUDA_BATCH_RESIDENT=0", all_run_flags=["+seed=3"])
test.file_grep(test.run_log_filename, r'seed=3 cyc=20 lfsr=1572864 time=205')

test.passes()
