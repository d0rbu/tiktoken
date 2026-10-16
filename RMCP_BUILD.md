# RMCP native owned-buffer batch build

This source starts at upstream tiktoken 0.14.0, commit
`4e71bbe0c078468e00fefbf94b39849389f346e5`, and extends the delivered RMCP anchored
matcher `a4804e19fb491d12e61bb06bfa635a989c158bbd`. Python distribution
version `0.14.0+rmcp.3` identifies the earlier owned-buffer delivery. The upstream MIT license is
retained.

The Rust sources, Cargo inputs and native tests are byte-identical to the measured
owned-buffer batch candidate, source snapshot
`e15c430714c4ac208f23ff576ffa37c808e9b294`. Only the Python version, wheel-test
dependency declaration and this build note change for delivery. Cargo's package
version remains `0.14.0` so its tested lockfile is preserved.

For the exact frozen byte-level pattern in `src/rmcp_pattern.txt`, the matcher
uses an anchored regular-expression search from the known piece boundary. It
retains noncapturing borrowed spans, the exact whitespace-suffix classification,
full-alternative restart after withholding the final whitespace scalar, match
order, UTF-8 boundaries and each special-delimited segment's end. The matcher
configuration, feature union and 128 per-thread matcher clones are preserved.
Every other pattern uses the original fancy-regex implementation. The public
Python `Encoding` interface, special-token handling and BPE merging are unchanged.

## Earlier anchored-matcher evidence

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
recorded in the delivery source receipt outside this repository. These historical
measurements concern the anchored matcher; the owned-buffer batch comparison is
reported separately below.

## Build

The patched distribution requires Python 3.10 or newer because its pinned build
backend no longer supports Python 3.9. RMCP itself uses Python 3.12.

Install Rust through rustup so `rust-toolchain.toml` selects Rust 1.90.0. The
committed Cargo lock is included in source distributions and enforced for every
Cargo command issued by setuptools-rust. The Python build requirements are
pinned in `pyproject.toml`; wheel builds use the release profile.

The cibuildwheel test setup declares NumPy alongside pytest and Hypothesis for
the owned-buffer extension tests. NumPy is a test-only dependency in this delivery;
the package's runtime dependencies are unchanged.

```sh
env SOURCE_DATE_EPOCH=1791355200 RUSTFLAGS= CARGO_ENCODED_RUSTFLAGS= \
  uv build --sdist --python 3.12.11
env SOURCE_DATE_EPOCH=1791355200 RUSTFLAGS= CARGO_ENCODED_RUSTFLAGS= \
  uv build --wheel --python 3.12.11
cargo test --locked --release --lib -- --test-threads=1
```

Source installs require the native compiler toolchain. Generated wheels, build
outputs and local experiment receipts do not belong in Git. Select this
distribution by its full immutable Git commit, rather than a floating branch.
Source and toolchain pinning makes the build inputs explicit; it is not a claim
of byte-identical wheels across build hosts or platforms.

## Serial owned-buffer batch contract and measured screen

This candidate starts at exact delivery `a4804e19fb491d12e61bb06bfa635a989c158bbd`.
It adds only the private native method
`CoreBPE.encode_to_tiktoken_buffers(texts, allowed_special)`. The return is a
Python list of the existing readonly `TiktokenBuffer` owners, one for each input
document, including empty documents. The public Python `Encoding` API remains
unchanged. The comparison used separately identified `0.14.0+rmcp.2` experimental
wheel overlays; this `0.14.0+rmcp.3` delivery retains their candidate implementation.

PyO3 retains each immutable input through `PyBackedStr`; the serial Rust loop
borrows its UTF-8 bytes rather than creating an additional owned `String` per
document. Python's UTF-8 conversion/cache and the vector of Python-backed input
owners still exist. One detached region and one allowed-special conversion feed
the unchanged scalar `CoreBPE::encode` in input order. The first encoding error
stops the loop and uses the scalar binding's `ValueError` message mapping. No
later document is encoded, no partial result is returned, and no retry occurs.
Each output `Vec<Rank>` moves into its own original buffer owner. No texts are
concatenated, and no token-ID lists, worker threads, cache or matcher changes
are introduced.

The private caller must retain whole-frame surrogate preflight and HF fallback,
then form NumPy views and invoke the original warnings only after all native
encodes complete. Batch argument extraction occurs before encoding; buffer
wrapping occurs after all encodes. Resource/allocation failure timing is not
claimed identical to repeated scalar calls. This API itself imposes no batch
size limit; RMCP retains its existing 64-document/1 MiB cuts and oversized
singleton behavior.

The complete native Rust `--lib` suite passed all 16 cases, including three new
controls for scalar parity/boundaries, independent owned vectors and first-error
short-circuiting. Seven Python extension cases in
`tests/test_owned_batch.py` cover actual buffer bytes, readonly/lifetime behavior,
empty inputs and strict text extraction without registry access or downloads.
All seven passed in the candidate's 60-case focused RMCP run; the baseline passed
the 53 common cases. The API and caller contract received independent source review.

The fixed A/B/B/A complete-command screen used CPython 3.12.11 on Linux x86-64,
16 workers per arm, and the same 2,880-shard input containing 1,356,564,136 IDs.
A used RMCP `530bff39b48f176be7b4b27448f601cfb8caac1e` with the retained rebuilt
`a4804e1` native wheel. B combined this native batch API with the reviewed RMCP
bridge, content snapshot `cd5d66ba23caf3505aa364ee1ee8f083d51602ea`.
Pooled whole-command throughput increased 6.4993% (ratio 1.064993); the opposite-order
pair ratios were 1.059142 and 1.071235. Pooled child CPU per ID had ratio 1.004622.
The fixed screen rules passed, and all four outputs matched every byte of the
retained 84-file artifact and complete manifest. Original producers, verifiers
and duplicate cleanup all closed; independent saved-result review accepted the
comparison. Evidence remains under `native-owned-batch-screen-v1` in the external
`packing-document-shards-mainb448-20261010-v1` evidence root.

This measures the combined native API and RMCP bridge on one fixed shared-host
cohort, including command startup and publication. It does not establish isolated
native-call speedup, 500B-token throughput, GPU behavior or full RMCP qualification.
The new-version source artifact still requires an ordinary build/install, its
native extension controls and the exact locked RMCP integration gate before
adoption; the measured wheel overlays do not satisfy that delivery qualification.


## Direct forward lazy-DFA delivery

Version `0.14.0+rmcp.4` retains the owned-buffer API and starts from exact native
parent `1ac68e9f91c5bac465d1557fbf18ad092709d28c`. For the existing exact guarded
pattern only, a forward lazy DFA replaces the meta-regex strategy dispatch and
reverse search. The next match start is already known. The NFA keeps UTF-8,
leftmost-first semantics, the 10 MiB construction limit and the 2 MiB lazy cache.
Every other pattern still takes the original fancy-regex route.

The whitespace suffix correction still withholds the final scalar when needed
and restarts all alternatives there, including contractions. Each ordinary segment
retains its own EOF before an allowed special. BPE, special matching, Python
buffers and public APIs are unchanged. Each iterator holds a cache-pool guard;
nested/concurrent iterators own separate caches, and clones share the immutable
DFA while owning separate pools. Cargo inputs, lock and toolchain are unchanged.

All 16 previous Rust semantic controls were retained, with only private matcher
type/capture-storage adaptations. Two additional controls passed: every one of
1,112,064 Unicode scalars with surrounding context, and nested/early-close/clone/
concurrent cache use. Raw API comparison passed 1,587 exact outcomes (including
matching negative outcomes), 512 positive real-document token/decode comparisons
and 512 concurrent comparisons. Unstable completions were compared as exact sets
because their ordering is unspecified.

The fixed A1/B1/B2/A2 screen used identical RMCP `983421f8` producer source,
CPython 3.12.11, a Ryzen 9950X, 16 processes and one native thread per process.
Each fresh command processed 737,248 train documents and 1,356,564,136 stored IDs,
including an unsplit 8,595,449-byte document. Command times were 65.215834 and
67.215237 seconds for the rebuilt parent, and 54.096739 and 59.630949 seconds for
the candidate. Pooled throughput increased 16.4458%; aggregate self/child CPU fell
17.1334%. Both temporal pairs improved and both arms met the declared 1.15
max/min timing-spread ceiling. All 84 complete output files matched byte for byte
in every run; warnings and actual worker native-library mappings also matched.

These are two matched pairs on one shared-host corpus with uncontrolled cache
residency. The producer snapshot used by both arms was explicitly unqualified;
this screen isolates the native delta and does not qualify that producer. Native
libraries were controlled private overlays, not release wheel installations.
Release build/install, current-main RMCP comparison and the final locked local
gate remain separate requirements. No GPU or 500B-token throughput claim follows.
Retained evidence is `pretrain-lean-native-mainf813-20261016-v1/`
`native-forward-dfa-full-panel-v1`; closure SHA-256 is
`26bf5f1092895bdaf22e5b85f34c297559cbc0727b48128a1cd47dcc24ba54a2`.
The earlier build-directory reuse error and rejected lean-worker timing evidence
remain retained separately; they were not used as passing measurements.
