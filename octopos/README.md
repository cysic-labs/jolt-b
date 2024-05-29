# README

OctoPos(eidon) Merkle Tree Implementation

This is a Merkle Tree implementation introduced specifically for Goldilocks
and Poseidon hash.  In this tree, we optimized first layer, that aggregates
eight Goldilocks field elements into one Poseidon hash for one hash digest.
The Poseidon hash should be parameterized with input width being 12 field
elements while the output width should be 4 field elements.

Benchmark on components of MT follows, rayon might add some overhead to each
sub-task appearing in each thread:

```
poseidion leaves/poseidon leaves/10
                        time:   [1.6516 ms 1.6523 ms 1.6532 ms]
                        change: [-12.188% -12.033% -11.886%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 5 outliers among 100 measurements (5.00%)
  2 (2.00%) low mild
  2 (2.00%) high mild
  1 (1.00%) high severe
poseidion leaves/poseidon leaves/11
                        time:   [2.1737 ms 2.1760 ms 2.1790 ms]
                        change: [+0.2828% +0.5652% +0.7863%] (p = 0.00 < 0.05)
                        Change within noise threshold.
poseidion leaves/poseidon leaves/12
                        time:   [2.8138 ms 2.9801 ms 3.0610 ms]
                        change: [+2.2737% +5.6974% +9.7220%] (p = 0.01 < 0.05)
                        Performance has regressed.
poseidion leaves/poseidon leaves/13
                        time:   [4.1606 ms 4.2130 ms 4.2879 ms]
                        change: [-4.2830% -1.6073% +1.2131%] (p = 0.28 > 0.05)
                        No change in performance detected.
poseidion leaves/poseidon leaves/14
                        time:   [6.9631 ms 6.9690 ms 6.9763 ms]
                        change: [-11.168% -10.313% -9.5985%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 2 outliers among 10 measurements (20.00%)
  2 (20.00%) high mild
poseidion leaves/poseidon leaves/15
                        time:   [13.176 ms 13.439 ms 13.651 ms]
                        change: [-5.5192% -4.1617% -3.1188%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 3 outliers among 10 measurements (30.00%)
  2 (20.00%) low severe
  1 (10.00%) high severe
poseidion leaves/poseidon leaves/16
                        time:   [24.179 ms 24.216 ms 24.252 ms]
                        change: [-8.7157% -8.5194% -8.3453%] (p = 0.00 < 0.05)
                        Performance has improved.
poseidion leaves/poseidon leaves/17
                        time:   [49.624 ms 49.673 ms 49.746 ms]
                        change: [-3.8870% -3.5556% -3.2405%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild
poseidion leaves/poseidon leaves/18
                        time:   [92.577 ms 95.935 ms 100.93 ms]
                        change: [-8.2736% -5.9198% -2.8987%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe
poseidion leaves/poseidon leaves/19
                        time:   [199.51 ms 200.11 ms 200.74 ms]
                        change: [-1.1811% -0.6474% -0.1412%] (p = 0.03 < 0.05)
                        Change within noise threshold.
poseidion leaves/poseidon leaves/20
                        time:   [362.52 ms 362.90 ms 363.72 ms]
                        change: [-11.306% -10.335% -9.5607%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild

poseidion internals/poseidon internals/10
                        time:   [2.9220 ms 2.9235 ms 2.9249 ms]
                        change: [-0.5589% +0.0299% +0.5908%] (p = 0.92 > 0.05)
                        No change in performance detected.
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
poseidion internals/poseidon internals/11
                        time:   [4.3073 ms 4.3202 ms 4.3336 ms]
                        change: [+1.5287% +1.7705% +1.9708%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) low mild
poseidion internals/poseidon internals/12
                        time:   [7.3221 ms 7.4978 ms 7.5975 ms]
                        change: [-8.0953% -7.1726% -6.0152%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 2 outliers among 10 measurements (20.00%)
  2 (20.00%) high mild
poseidion internals/poseidon internals/13
                        time:   [13.200 ms 13.262 ms 13.315 ms]
                        change: [-8.5053% -7.9892% -7.5639%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) low severe
poseidion internals/poseidon internals/14
                        time:   [24.772 ms 24.778 ms 24.789 ms]
                        change: [-6.8058% -6.4745% -6.1179%] (p = 0.00 < 0.05)
                        Performance has improved.
poseidion internals/poseidon internals/15
                        time:   [48.579 ms 48.926 ms 49.599 ms]
                        change: [-0.4588% +0.1857% +0.9341%] (p = 0.63 > 0.05)
                        No change in performance detected.
poseidion internals/poseidon internals/16
                        time:   [102.38 ms 103.68 ms 104.19 ms]
                        change: [-0.8480% +2.1790% +4.9419%] (p = 0.16 > 0.05)
                        No change in performance detected.
poseidion internals/poseidon internals/17
                        time:   [207.16 ms 209.01 ms 212.40 ms]
                        change: [+7.2454% +8.1927% +9.4511%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild
poseidion internals/poseidon internals/18
                        time:   [411.61 ms 412.89 ms 414.30 ms]
                        change: [+7.7068% +7.9230% +8.1755%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe
poseidion internals/poseidon internals/19
                        time:   [765.84 ms 810.09 ms 843.88 ms]
                        change: [-7.9385% -5.5942% -2.5055%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 3 outliers among 10 measurements (30.00%)
  1 (10.00%) low mild
  2 (20.00%) high severe
Benchmarking poseidion internals/poseidon internals/20: Warming up for 3.0000 s
Warning: Unable to complete 10 samples in 50.0s. You may wish to increase target time to 83.0s or enable flat sampling.
poseidion internals/poseidon internals/20
                        time:   [1.5074 s 1.5081 s 1.5091 s]
                        change: [-13.143% -13.001% -12.849%] (p = 0.00 < 0.05)
                        Performance has improved.

octopos tree/octopos MT/10
                        time:   [256.46 µs 257.67 µs 259.01 µs]
                        change: [-5.7027% -3.8302% -2.1609%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 10 outliers among 100 measurements (10.00%)
  3 (3.00%) low mild
  4 (4.00%) high mild
  3 (3.00%) high severe
octopos tree/octopos MT/11
                        time:   [404.84 µs 407.04 µs 408.55 µs]
                        change: [-5.1551% -4.1719% -3.3146%] (p = 0.00 < 0.05)
                        Performance has improved.
octopos tree/octopos MT/12
                        time:   [698.27 µs 704.41 µs 711.31 µs]
                        change: [-8.0574% -5.5160% -3.1681%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe
octopos tree/octopos MT/13
                        time:   [1.2576 ms 1.2667 ms 1.2746 ms]
                        change: [-12.161% -8.8185% -5.8971%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild
octopos tree/octopos MT/14
                        time:   [2.3434 ms 2.3502 ms 2.3553 ms]
                        change: [-10.888% -8.8573% -7.1017%] (p = 0.00 < 0.05)
                        Performance has improved.
octopos tree/octopos MT/15
                        time:   [4.4765 ms 4.4890 ms 4.4972 ms]
                        change: [-7.4140% -7.0223% -6.5566%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild
octopos tree/octopos MT/16
                        time:   [8.6862 ms 8.7046 ms 8.7310 ms]
                        change: [-8.4316% -6.7800% -5.5818%] (p = 0.00 < 0.05)
                        Performance has improved.
octopos tree/octopos MT/17
                        time:   [17.076 ms 17.144 ms 17.229 ms]
                        change: [-11.549% -8.9458% -6.4813%] (p = 0.00 < 0.05)
                        Performance has improved.
octopos tree/octopos MT/18
                        time:   [33.494 ms 33.616 ms 33.811 ms]
                        change: [-14.456% -12.543% -10.851%] (p = 0.00 < 0.05)
                        Performance has improved.
octopos tree/octopos MT/19
                        time:   [66.733 ms 66.847 ms 66.979 ms]
                        change: [-11.169% -10.523% -9.9189%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 2 outliers among 10 measurements (20.00%)
  1 (10.00%) low severe
  1 (10.00%) high mild
octopos tree/octopos MT/20
                        time:   [135.88 ms 136.32 ms 136.68 ms]
                        change: [-7.2719% -6.4588% -5.6333%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high mild
octopos tree/octopos MT/21
                        time:   [268.30 ms 269.50 ms 270.64 ms]
                        change: [-8.6047% -7.5667% -6.4765%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 2 outliers among 10 measurements (20.00%)
  1 (10.00%) low mild
  1 (10.00%) high mild
Benchmarking octopos tree/octopos MT/22: Warming up for 3.0000 s
Warning: Unable to complete 10 samples in 20.0s. You may wish to increase target time to 29.5s or enable flat sampling.
octopos tree/octopos MT/22
                        time:   [534.89 ms 536.22 ms 538.28 ms]
                        change: [-19.338% -13.847% -8.3797%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 10 measurements (10.00%)
  1 (10.00%) high severe
octopos tree/octopos MT/23
                        time:   [1.0650 s 1.0683 s 1.0715 s]
                        change: [-22.483% -17.963% -13.370%] (p = 0.00 < 0.05)
                        Performance has improved.
Benchmarking octopos tree/octopos MT/24: Warming up for 3.0000 s
Warning: Unable to complete 10 samples in 20.0s. You may wish to increase target time to 21.7s.
octopos tree/octopos MT/24
                        time:   [2.1318 s 2.1382 s 2.1447 s]
                        change: [-10.915% -9.4335% -7.8705%] (p = 0.00 < 0.05)
                        Performance has improved.
Benchmarking octopos tree/octopos MT/25: Warming up for 3.0000 s
Warning: Unable to complete 10 samples in 20.0s. You may wish to increase target time to 44.3s.
octopos tree/octopos MT/25
                        time:   [4.2621 s 4.2741 s 4.2869 s]
                        change: [-10.617% -9.6486% -8.6576%] (p = 0.00 < 0.05)
                        Performance has improved.
Benchmarking octopos tree/octopos MT/26: Warming up for 3.0000 s
Warning: Unable to complete 10 samples in 20.0s. You may wish to increase target time to 89.0s.
octopos tree/octopos MT/26
                        time:   [8.5578 s 8.5710 s 8.5863 s]
                        change: [-7.1292% -6.0480% -5.0824%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 2 outliers among 10 measurements (20.00%)
  2 (20.00%) high mild
Benchmarking octopos tree/octopos MT/27: Warming up for 3.0000 s
Warning: Unable to complete 10 samples in 20.0s. You may wish to increase target time to 174.6s.
octopos tree/octopos MT/27
                        time:   [17.101 s 17.150 s 17.195 s]
                        change: [-5.3919% -4.9130% -4.4774%] (p = 0.00 < 0.05)
                        Performance has improved.
```
