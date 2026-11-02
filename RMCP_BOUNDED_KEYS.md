# Bounded whole-piece rank lookup

The guarded RMCP regex path batches complete ASCII piece boundaries into a
64-byte bitset. Once a piece boundary is known, a one-to-seven-byte piece can
use one little-endian eight-byte load and the existing length-tagged rank map.
The load runs only when eight bytes remain inside the current ordinary segment;
masking excludes every byte beyond the piece. Short tails use the original lookup.
There is no new rank table or vocabulary cache.

The vector scanner retains an AVX2 runtime guard. Non-ASCII boundaries, custom
patterns, specials, missing ranks, count-only calls and unsupported hardware keep
their existing paths. Token IDs, lengths, order and packed artifact formats do not
change. The new tests compare exact boundaries and ranks across alignment, tails,
Unicode, embedded zero bytes and extreme rank values.

## Correctness and source binding

- Baseline: native `94b29f00c67bc3307d51c0a1e89d6b40f5d2f5c5`, the required
  dependency of RMCP `9007f04a4481fddfb10d1d136f68c6e46d451cf2`.
- Integrating native main `266e2c89df373ecb5c271c9d20d259430caf3fc4` produces
  `a5385e9793eb5b10c4ddf1f15904cfb2f417c673`, with exactly the same baseline tree
  `bf8d7f3c35859267f12c19a78e33f067c834b8f5`. All compiled Rust and Python
  source bytes were checked against the compiled sources.
- Rust 1.90.0 release builds, identical flags/features; 60 baseline and 66 candidate
  Rust unit tests passed. The integrated source passed all 2,576 Python tests and
  17,160 independent original-library comparisons under CPython 3.12.11.
- All 27 preparation commands succeeded. Every complete artifact matched in
  literal bytes through EOF: token files, document offsets and complete manifest.
  Warning multisets also matched. No numerical tolerance was needed.

## Complete-command measurement

AMD 9950X, affinity CPUs 0-15, 16 source workers, inner library threads one. Both
arms use the existing RMCP large-shard recipe: packed storage and 268,435,456
tokens per shard. Timers include Python startup, source reads, decode, tokenization,
IPC, packing, file sync, publication and process exit. Verification is outside
the timer. This changes only the native library between A and B.

Each command repeats the same 135 real Python/SQL/PDF/web source files 32 times:
34,442 distinct documents, 1,102,144 document visits and 1,032,005,280 token IDs.
Page cache is warm/shared. This is not a unique-billion dataset or a cold-storage
500B-token result. The exact input manifest SHA256 is
`c82a440bf7409e97a8c6f0065943f287c4eb5311a13719f125a00b83ae8aab87`.

Three unscored warm controls precede 12 fixed adjacent pairs. Pair order is
BA, AB, BA, AB, BA, AB, AB, BA, BA, AB, BA, AB. No sample is dropped or replaced.
A seeded 10,000-draw bootstrap resamples six adjacent AB/BA pair blocks.

| Measure | Result |
| --- | --- |
| Geometric whole-command A/B | 1.062389 |
| 95% paired-block interval | 1.020673-1.105845 |
| Faster B pairs | 10 / 12 |
| Geometric CPU B/A | 0.835695 |
| CPU 95% paired-block interval | 0.808812-0.862203 |
| CPU wins | 12 / 12 |
| Warm repeated-fixture A rate | 108.435M IDs/s |
| Warm repeated-fixture B rate | 115.200M IDs/s |

All preregistered thresholds passed: speedup at least 1.05, lower interval above
1.02, at least ten elapsed wins, CPU ratio at most 0.95, CPU upper interval below
0.99, at least ten CPU wins and improvements in both order strata. Raw elapsed
spreads remain visible: A 1.333x, B 1.346x. The interval lower bound only narrowly
clears its threshold; broader workload/hardware claims require more evidence.

| Pair | A elapsed s | B elapsed s | A CPU s | B CPU s |
| --- | --- | --- | --- | --- |
| 1 | 9.989935 | 10.492038 | 45.283027 | 41.795801 |
| 2 | 9.392797 | 9.187693 | 41.829483 | 35.851535 |
| 3 | 10.281634 | 9.193793 | 43.066603 | 35.473690 |
| 4 | 9.559894 | 9.341858 | 40.674233 | 34.904601 |
| 5 | 9.745542 | 8.993629 | 41.585866 | 34.058252 |
| 6 | 9.500016 | 9.036730 | 41.200449 | 34.459046 |
| 7 | 11.261101 | 9.148760 | 46.733550 | 35.708312 |
| 8 | 10.155012 | 9.292610 | 43.990209 | 35.100228 |
| 9 | 9.051057 | 9.501272 | 40.446254 | 34.736160 |
| 10 | 8.448018 | 7.796570 | 38.632672 | 32.131549 |
| 11 | 8.700900 | 8.064792 | 38.590292 | 32.126102 |
| 12 | 8.500945 | 7.816827 | 37.891345 | 31.661250 |

Unscored warm controls (elapsed/CPU seconds): W0 11.306740/50.045177, W1 11.775006/51.446620, WB 11.279062/43.735828.

## Adverse evidence and cleanup

Earlier same-candidate component and tiered/smaller-shard studies failed their
predeclared timing criteria. They remain failures, and their samples are not
pooled here. An attempted native-main-only comparison failed before token output
because RMCP requires the open native dependency stack's model-admission API;
the corrected dependency bridge above preserves it. An earlier Python test
harness passed 2,559 cases but failed two child imports; fresh interpreters had
not inherited its extension overlay. The corrected private package setup keeps
all assertions and passes the complete suite.

All 135 newly generated NPY bodies and 4,320 private source aliases were removed
after exact comparison and process closure. Headers, locks, source, environments,
measured libraries, reports and failure evidence remain. Peak charged memory was
4.961 GiB under 12 GiB with no memory-limit or OOM events. No
production environment or corpus was migrated.
