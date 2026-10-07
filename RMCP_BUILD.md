# RMCP native span build

This source starts at upstream tiktoken 0.14.0, commit
`4e71bbe0c078468e00fefbf94b39849389f346e5`. Version `0.14.0+rmcp.1`
identifies the patched distribution. The upstream MIT license is retained.

The native change uses noncapturing match spans for the exact frozen byte-level
pattern in `src/rmcp_pattern.txt`. It preserves whitespace suffix handling, match
order, UTF-8 boundaries and special-token segmentation. Every other pattern uses
the original fancy-regex implementation. Python encoding interfaces and BPE
merging are unchanged.

The RMCP qualification used CPython 3.12.11 on Linux x86-64. Native Unicode and
segmentation controls, 21,373 model cases, 173 targeted Python tests, and complete
2,924,801,238-ID packing outputs agreed exactly with stock. An eight-worker,
six-command comparison measured 21.74% more whole-command throughput than the
published upstream wheel and 25.53% more than a matched locally built stock
wheel. Those are separate comparisons on one selected corpus and shared host;
they do not establish portability or a full-corpus rate.

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
