### chunk_size=0

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                     aten::_log_softmax         7.35%      32.840ms         7.35%      32.840ms      32.840ms     500.00 MB     500.00 MB             1
                                aten::nll_loss_backward         0.41%       1.824ms        20.28%      90.546ms      90.546ms     500.00 MB     500.00 MB             1
                       aten::_log_softmax_backward_data        68.27%     304.859ms        68.27%     304.859ms     304.859ms     500.00 MB     500.00 MB             1
                                 aten::nll_loss_forward         0.25%       1.101ms         0.25%       1.101ms       1.101ms           8 B           8 B             1
                                    aten::empty_strided         0.00%       2.292us         0.00%       2.292us       2.292us           4 B           4 B             1
               <frozen runpy>(198): _run_module_as_main         0.00%       0.917us       100.00%     446.555ms     446.555ms     500.00 MB           0 B             1
                          <frozen runpy>(88): _run_code         0.00%       1.458us       100.00%     446.555ms     446.555ms     500.00 MB           0 B             1
        litgpt/scripts/profile_memory.py(125): <module>         0.00%       1.125us       100.00%     446.553ms     446.553ms     500.00 MB           0 B             1
             litgpt/scripts/profile_memory.py(70): main         0.00%       1.333us       100.00%     446.552ms     446.552ms     500.00 MB           0 B             1
litgpt/scripts/profile_memory.py(33): profile_chunk_...         0.01%      42.680us       100.00%     446.551ms     446.551ms     500.00 MB           0 B             1
            torch/profiler/profiler.py(1118): __enter__         0.00%       0.416us         0.01%      65.118us      65.118us           0 B           0 B             1
                torch/profiler/profiler.py(1128): start         0.00%       0.459us         0.01%      64.702us      64.702us           0 B           0 B             1
      torch/profiler/profiler.py(1262): _transit_action         0.00%       1.792us         0.01%      64.243us      64.243us           0 B           0 B             1
           torch/profiler/profiler.py(342): start_trace         0.00%      13.082us         0.01%      62.451us      62.451us           0 B           0 B             1
          torch/autograd/profiler.py(414): _start_trace         0.01%      29.392us         0.01%      30.892us      30.892us           0 B           0 B             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 446.555ms

```

### chunk_size=32

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::cat        40.86%     102.629ms        40.87%     102.635ms      51.318ms     500.02 MB     500.02 MB             2
                                     aten::_log_softmax        31.04%      77.958ms        31.04%      77.958ms     609.050us     500.00 MB     500.00 MB           128
                                aten::nll_loss_backward         0.69%       1.745ms         2.19%       5.488ms      42.873us     500.00 MB     500.00 MB           128
                       aten::_log_softmax_backward_data        14.33%      35.987ms        14.33%      35.987ms     281.146us     500.00 MB     500.00 MB           128
                                    aten::empty_strided         0.00%       8.894us         0.00%       8.894us       1.779us      32.02 KB      32.02 KB             5
                                 aten::nll_loss_forward         1.24%       3.107ms         1.24%       3.107ms      24.270us      16.50 KB      16.50 KB           128
                                               aten::ne         0.01%      21.167us         0.01%      21.167us      21.167us       4.00 KB       4.00 KB             1
                                          aten::maximum         0.00%       8.209us         0.00%       8.209us       8.209us           8 B           8 B             1
               <frozen runpy>(198): _run_module_as_main         0.00%       0.500us       100.00%     251.147ms     251.147ms     500.00 MB           0 B             1
                          <frozen runpy>(88): _run_code         0.00%       1.125us       100.00%     251.147ms     251.147ms     500.00 MB           0 B             1
        litgpt/scripts/profile_memory.py(125): <module>         0.00%       0.958us       100.00%     251.145ms     251.145ms     500.00 MB           0 B             1
             litgpt/scripts/profile_memory.py(70): main         0.00%       0.958us       100.00%     251.144ms     251.144ms     500.00 MB           0 B             1
            torch/profiler/profiler.py(1118): __enter__         0.00%       0.416us         0.05%     133.903us     133.903us           0 B           0 B             1
                torch/profiler/profiler.py(1128): start         0.00%       0.417us         0.05%     133.487us     133.487us           0 B           0 B             1
      torch/profiler/profiler.py(1262): _transit_action         0.00%       7.333us         0.05%     133.070us     133.070us           0 B           0 B             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 251.147ms

```

### chunk_size=64

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::cat         8.38%       6.984ms         8.38%       6.991ms       3.496ms     500.02 MB     500.02 MB             2
                                     aten::_log_softmax        44.11%      36.782ms        44.11%      36.782ms     574.718us     500.00 MB     500.00 MB            64
                                aten::nll_loss_backward         1.24%       1.030ms         4.69%       3.908ms      61.061us     500.00 MB     500.00 MB            64
                       aten::_log_softmax_backward_data        31.84%      26.553ms        31.84%      26.553ms     414.884us     500.00 MB     500.00 MB            64
                                    aten::empty_strided         0.00%       2.709us         0.00%       2.709us       0.542us      32.02 KB      32.02 KB             5
                                 aten::nll_loss_forward         1.30%       1.082ms         1.30%       1.082ms      16.908us      16.25 KB      16.25 KB            64
                                               aten::ne         0.01%      10.560us         0.01%      10.560us      10.560us       4.00 KB       4.00 KB             1
                                          aten::maximum         0.00%       2.583us         0.00%       2.583us       2.583us           8 B           8 B             1
               <frozen runpy>(198): _run_module_as_main         0.00%       0.625us       100.00%      83.393ms      83.393ms     500.00 MB           0 B             1
                          <frozen runpy>(88): _run_code         0.00%       1.334us       100.00%      83.392ms      83.392ms     500.00 MB           0 B             1
        litgpt/scripts/profile_memory.py(125): <module>         0.00%       0.875us       100.00%      83.391ms      83.391ms     500.00 MB           0 B             1
             litgpt/scripts/profile_memory.py(70): main         0.00%       1.166us       100.00%      83.390ms      83.390ms     500.00 MB           0 B             1
            torch/profiler/profiler.py(1118): __enter__         0.00%       0.292us         0.07%      56.870us      56.870us           0 B           0 B             1
                torch/profiler/profiler.py(1128): start         0.00%       0.292us         0.07%      56.578us      56.578us           0 B           0 B             1
      torch/profiler/profiler.py(1262): _transit_action         0.00%       2.125us         0.07%      56.286us      56.286us           0 B           0 B             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 83.393ms

```

### chunk_size=128

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::cat         4.28%       6.221ms         4.28%       6.229ms       3.114ms     500.02 MB     500.02 MB             2
                                     aten::_log_softmax        51.73%      75.246ms        51.73%      75.246ms       2.351ms     500.00 MB     500.00 MB            32
                                aten::nll_loss_backward         0.84%       1.227ms         4.21%       6.122ms     191.301us     500.00 MB     500.00 MB            32
                       aten::_log_softmax_backward_data        27.44%      39.912ms        27.44%      39.912ms       1.247ms     500.00 MB     500.00 MB            32
                                    aten::empty_strided         0.00%       4.916us         0.00%       4.916us       0.983us      32.02 KB      32.02 KB             5
                                 aten::nll_loss_forward         1.34%       1.950ms         1.34%       1.950ms      60.949us      16.12 KB      16.12 KB            32
                                               aten::ne         0.01%      10.458us         0.01%      10.458us      10.458us       4.00 KB       4.00 KB             1
                                          aten::maximum         0.00%       2.833us         0.00%       2.833us       2.833us           8 B           8 B             1
               <frozen runpy>(198): _run_module_as_main         0.00%       0.584us       100.00%     145.471ms     145.471ms     500.00 MB           0 B             1
                          <frozen runpy>(88): _run_code         0.00%       1.583us       100.00%     145.471ms     145.471ms     500.00 MB           0 B             1
        litgpt/scripts/profile_memory.py(125): <module>         0.00%       1.559us       100.00%     145.469ms     145.469ms     500.00 MB           0 B             1
             litgpt/scripts/profile_memory.py(70): main         0.00%       1.291us       100.00%     145.468ms     145.468ms     500.00 MB           0 B             1
            torch/profiler/profiler.py(1118): __enter__         0.00%       0.458us         0.04%      58.268us      58.268us           0 B           0 B             1
                torch/profiler/profiler.py(1128): start         0.00%       0.459us         0.04%      57.810us      57.810us           0 B           0 B             1
      torch/profiler/profiler.py(1262): _transit_action         0.00%       1.916us         0.04%      57.351us      57.351us           0 B           0 B             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 145.471ms

```

### chunk_size=256

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::cat         6.84%       5.861ms         6.85%       5.869ms       2.934ms     500.02 MB     500.02 MB             2
                                     aten::_log_softmax        44.01%      37.701ms        44.01%      37.701ms       2.356ms     500.00 MB     500.00 MB            16
                                aten::nll_loss_backward         0.59%     504.330us         5.91%       5.065ms     316.553us     500.00 MB     500.00 MB            16
                       aten::_log_softmax_backward_data        37.91%      32.479ms        37.91%      32.479ms       2.030ms     500.00 MB     500.00 MB            16
                                    aten::empty_strided         0.00%       3.209us         0.00%       3.209us       0.642us      32.02 KB      32.02 KB             5
                                 aten::nll_loss_forward         0.76%     648.975us         0.76%     648.975us      40.561us      16.06 KB      16.06 KB            16
                                               aten::ne         0.02%      14.060us         0.02%      14.060us      14.060us       4.00 KB       4.00 KB             1
                                          aten::maximum         0.00%       2.666us         0.00%       2.666us       2.666us           8 B           8 B             1
               <frozen runpy>(198): _run_module_as_main         0.00%       0.642us       100.00%      85.670ms      85.670ms     500.00 MB           0 B             1
                          <frozen runpy>(88): _run_code         0.00%       1.791us       100.00%      85.669ms      85.669ms     500.00 MB           0 B             1
        litgpt/scripts/profile_memory.py(125): <module>         0.00%       0.917us       100.00%      85.667ms      85.667ms     500.00 MB           0 B             1
             litgpt/scripts/profile_memory.py(70): main         0.00%       1.167us       100.00%      85.666ms      85.666ms     500.00 MB           0 B             1
            torch/profiler/profiler.py(1118): __enter__         0.00%       0.292us         0.06%      49.893us      49.893us           0 B           0 B             1
                torch/profiler/profiler.py(1128): start         0.00%       0.500us         0.06%      49.601us      49.601us           0 B           0 B             1
      torch/profiler/profiler.py(1262): _transit_action         0.00%       3.125us         0.06%      49.101us      49.101us           0 B           0 B             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 85.670ms

```

### chunk_size=512

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::cat         6.20%       6.522ms         6.21%       6.529ms       3.265ms     500.02 MB     500.02 MB             2
                                     aten::_log_softmax        44.91%      47.232ms        44.91%      47.232ms       5.904ms     500.00 MB     500.00 MB             8
                                aten::nll_loss_backward         0.53%     557.234us         8.70%       9.147ms       1.143ms     500.00 MB     500.00 MB             8
                       aten::_log_softmax_backward_data        35.12%      36.936ms        35.12%      36.936ms       4.617ms     500.00 MB     500.00 MB             8
                                    aten::empty_strided         0.00%       3.208us         0.00%       3.208us       0.642us      32.02 KB      32.02 KB             5
                                 aten::nll_loss_forward         1.25%       1.319ms         1.25%       1.319ms     164.870us      16.03 KB      16.03 KB             8
                                               aten::ne         0.01%      11.250us         0.01%      11.250us      11.250us       4.00 KB       4.00 KB             1
                                          aten::maximum         0.00%       2.542us         0.00%       2.542us       2.542us           8 B           8 B             1
               <frozen runpy>(198): _run_module_as_main         0.00%       0.583us       100.00%     105.179ms     105.179ms     500.00 MB           0 B             1
                          <frozen runpy>(88): _run_code         0.00%       1.417us       100.00%     105.179ms     105.179ms     500.00 MB           0 B             1
        litgpt/scripts/profile_memory.py(125): <module>         0.00%       2.100us       100.00%     105.177ms     105.177ms     500.00 MB           0 B             1
             litgpt/scripts/profile_memory.py(70): main         0.00%       1.209us       100.00%     105.175ms     105.175ms     500.00 MB           0 B             1
            torch/profiler/profiler.py(1118): __enter__         0.00%       0.375us         0.05%      50.226us      50.226us           0 B           0 B             1
                torch/profiler/profiler.py(1128): start         0.00%       0.292us         0.05%      49.851us      49.851us           0 B           0 B             1
      torch/profiler/profiler.py(1262): _transit_action         0.00%       1.957us         0.05%      49.559us      49.559us           0 B           0 B             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 105.179ms

```
