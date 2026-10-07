# RMCP native anchored-span build

This source starts at upstream tiktoken 0.14.0, commit
`4e71bbe0c078468e00fefbf94b39849389f346e5`, and extends the delivered RMCP span
implementation `313a3959314987bbb6130b94a25f85c202462f69`. Python distribution
version `0.14.0+rmcp.2` identifies this delivery. The upstream MIT license is
retained.

The Rust sources and Cargo inputs are byte-identical to the measured anchored
prototype `f62df98a1f3eab00751022c0d7278439444cff69`. Only the Python version and
this build note change for delivery. Cargo's package version remains `0.14.0`
so its tested lockfile is preserved.

For the exact frozen byte-level pattern in `src/rmcp_pattern.txt`, the matcher
uses an anchored regular-expression search from the known piece boundary. It
retains noncapturing borrowed spans, the exact whitespace-suffix classification,
full-alternative restart after withholding the final whitespace scalar, match
order, UTF-8 boundaries and each special-delimited segment's end. The matcher
configuration, feature union and 128 per-thread matcher clones are preserved.
Every other pattern uses the original fancy-regex implementation. Python
encoding interfaces, special-token handling and BPE merging are unchanged.

## Measured prototype scope

The experiment used fixed RMCP main
`6b4ae00487812e85d51d2a13f21b28a1d863cce0`, CPython 3.12.11, Linux x86-64 and
eight packing workers. Both native variants were locally built with the same
Rust/compiler and Python frontend settings and experimental version
`0.14.0+rmcp.1`. The comparator was the delivered span source `313a3959`; the
candidate was anchored source `f62df98a`, not a different published wheel.

The A1/C1/C2/A2 complete-command comparison processed 2,924,801,238 IDs per
command. All four outputs matched every byte of the retained 177-file cache
and its complete manifest. Candidate pooled throughput was 37.367546 million
IDs/s versus 34.476792 million for the matched baseline: 8.38% higher throughput.
The two temporal-pair wall-time ratios were 1.081758 and 1.085926; whole-cgroup
CPU per ID was 9.08% lower (candidate/baseline ratio 0.909233). These passed the
predeclared 5% pooled-rate, 3% each-pair and at-most-1.05 CPU-ratio screen.
These are one selected corpus, fixed order and shared-host measurements, not
statistical, cross-platform or full-corpus throughput claims.

Native controls passed 9 baseline and 11 candidate cases. Both arms passed
21,373 model cases and 173 targeted Python tests with identical IDs and packed
bytes. The original outer control wrapper failed only its execution-order versus
sorted-name metadata assertion; that failure is retained, with separate
independent acceptance of the exact passing child results. It is not reported
as a successful wrapper run.

The retained evidence is `packing-native-anchored-main6b4a-20261007-v1`:
`closeout-v1.json`, `comparison-archive-v1/independent-terminal-review-v1.json`
and `independent-correctness-review-v1.json`. Their immutable identities are
recorded in the delivery source receipt outside this repository. This packaging
step has not built or installed `0.14.0+rmcp.2`; its ordinary source build,
installation and exact RMCP integration gate remain required before adoption.

## Build

The patched distribution requires Python 3.10 or newer because its pinned build
backend no longer supports Python 3.9. RMCP itself uses Python 3.12.

Install Rust through rustup so `rust-toolchain.toml` selects Rust 1.90.0. The
committed Cargo lock is included in source distributions and enforced for every
Cargo command issued by setuptools-rust. The Python build requirements are
pinned in `pyproject.toml`; wheel builds use the release profile.

```sh
env SOURCE_DATE_EPOCH=1791355200 RUSTFLAGS= CARGO_ENCODED_RUSTFLAGS= \
  uv build --sdist --python 3.12.11
env SOURCE_DATE_EPOCH=1791355200 RUSTFLAGS= CARGO_ENCODED_RUSTFLAGS= \
  uv build --wheel --python 3.12.11
cargo test --locked --release --lib rmcp_regex::tests -- --test-threads=1
```

Source installs require the native compiler toolchain. Generated wheels, build
outputs and local experiment receipts do not belong in Git. Select this
distribution by its full immutable Git commit, rather than a floating branch.
Source and toolchain pinning makes the build inputs explicit; it is not a claim
of byte-identical wheels across build hosts or platforms.
