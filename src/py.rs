use std::collections::HashSet;

use pyo3::{
    IntoPyObjectExt, PyResult, exceptions,
    prelude::*,
    pybacked::PyBackedStr,
    types::{PyBytes, PyList},
};
use rustc_hash::FxHashMap as HashMap;

use crate::rmcp_source::{Batch as SourceBatch, Cursor as SourceCursor, ReadError};
use crate::rmcp_token_storage;
use crate::rmcp_token_tier;
use crate::{_byte_pair_encode, CoreBPE, Rank};

fn packed_ranks(mut raw: &[u8]) -> PyResult<HashMap<Vec<u8>, Rank>> {
    let mut ranks = HashMap::default();
    while !raw.is_empty() {
        let header = raw
            .get(..8)
            .ok_or_else(|| exceptions::PyValueError::new_err("truncated packed rank header"))?;
        let rank = u32::from_le_bytes(header[..4].try_into().unwrap());
        let size = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
        raw = &raw[8..];
        let word = raw
            .get(..size)
            .ok_or_else(|| exceptions::PyValueError::new_err("truncated packed rank word"))?;
        ranks.insert(word.to_vec(), rank);
        raw = &raw[size..];
    }
    if ranks.is_empty() {
        return Err(exceptions::PyValueError::new_err("empty packed ranks"));
    }
    Ok(ranks)
}

fn storage_width(width: u8) -> PyResult<u8> {
    match width {
        8..=32 => Ok(width),
        _ => Err(exceptions::PyValueError::new_err(
            "token storage width must be between 8 and 32",
        )),
    }
}

#[pyclass(name = "_TokenStorageEncoder")]
struct TokenStorageEncoder {
    encoder: rmcp_token_storage::Encoder,
}

#[pymethods]
impl TokenStorageEncoder {
    #[new]
    fn new(width: u8) -> PyResult<Self> {
        Ok(Self {
            encoder: rmcp_token_storage::Encoder::new(storage_width(width)?),
        })
    }

    fn push(&mut self, py: Python<'_>, raw: &Bound<'_, PyBytes>) -> Py<PyBytes> {
        let raw = raw.as_bytes();
        let bytes = py.detach(|| self.encoder.push(raw));
        PyBytes::new(py, bytes).unbind()
    }

    fn finish(&mut self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, self.encoder.finish()).unbind()
    }
}

#[pyclass(name = "_TokenStorageDecoder", frozen)]
struct TokenStorageDecoder {
    width: u8,
}

#[pyclass(name = "_TokenTierCodec", frozen)]
struct TokenTierCodec {
    codec: rmcp_token_tier::Codec,
}

#[pymethods]
impl TokenTierCodec {
    #[new]
    fn new(order: Vec<u32>, a: u8, b: u8) -> PyResult<Self> {
        let codec = rmcp_token_tier::Codec::new(&order, a, b).ok_or_else(|| {
            exceptions::PyValueError::new_err("invalid common-token tier codebook")
        })?;
        Ok(Self { codec })
    }

    fn encode(&self, py: Python<'_>, raw: &Bound<'_, PyBytes>) -> PyResult<Py<PyBytes>> {
        let raw = raw.as_bytes();
        let encoded = py
            .detach(|| self.codec.encode(raw))
            .ok_or_else(|| exceptions::PyValueError::new_err("invalid uint32 token block"))?;
        Ok(PyBytes::new(py, &encoded).unbind())
    }

    fn decode(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        count: usize,
    ) -> PyResult<Py<PyAny>> {
        let data = data.as_bytes();
        let decoded = py
            .detach(|| self.codec.decode(data, count))
            .ok_or_else(|| {
                exceptions::PyValueError::new_err("invalid prefix block extent or padding")
            })?;
        TiktokenBuffer::new(decoded).into_py_any(py)
    }
    fn encode_indexed(
        &self,
        py: Python<'_>,
        raw: &Bound<'_, PyBytes>,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        let raw = raw.as_bytes();
        let (data, checkpoints) = py
            .detach(|| self.codec.encode_indexed(raw))
            .ok_or_else(|| exceptions::PyValueError::new_err("invalid uint32 token block"))?;
        Ok((
            PyBytes::new(py, &data).unbind(),
            PyBytes::new(py, &checkpoints).unbind(),
        ))
    }

    fn decode_range(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        checkpoints: &Bound<'_, PyBytes>,
        tokens: usize,
        start: usize,
        count: usize,
    ) -> PyResult<Py<PyAny>> {
        let data = data.as_bytes();
        let checkpoints = checkpoints.as_bytes();
        let decoded = py
            .detach(|| {
                self.codec
                    .decode_range(data, checkpoints, tokens, start, count)
            })
            .ok_or_else(|| exceptions::PyValueError::new_err("invalid indexed token range"))?;
        TiktokenBuffer::new(decoded).into_py_any(py)
    }

    fn decode_ranges(
        &self,
        py: Python<'_>,
        ranges: Vec<(Py<PyBytes>, Py<PyBytes>, usize, usize, usize)>,
    ) -> PyResult<Py<PyAny>> {
        let borrowed: Vec<_> = ranges
            .iter()
            .map(
                |(data, checkpoints, tokens, start, count)| rmcp_token_tier::IndexedRange {
                    data: data.bind(py).as_bytes(),
                    checkpoints: checkpoints.bind(py).as_bytes(),
                    tokens: *tokens,
                    start: *start,
                    count: *count,
                },
            )
            .collect();
        let decoded = py
            .detach(|| self.codec.decode_ranges(&borrowed))
            .ok_or_else(|| exceptions::PyValueError::new_err("invalid indexed token range"))?;
        TiktokenBuffer::new(decoded).into_py_any(py)
    }

    fn encode_blocks(
        &self,
        py: Python<'_>,
        raw: &Bound<'_, PyBytes>,
        block_tokens: usize,
    ) -> PyResult<Vec<(Py<PyBytes>, Py<PyBytes>)>> {
        if block_tokens == 0 || block_tokens.checked_mul(4).is_none() {
            return Err(exceptions::PyValueError::new_err(
                "invalid token block size",
            ));
        }
        let raw = raw.as_bytes();
        let blocks = py
            .detach(|| self.codec.encode_blocks(raw, block_tokens))
            .ok_or_else(|| exceptions::PyValueError::new_err("invalid uint32 token block"))?;
        Ok(blocks
            .into_iter()
            .map(|(data, positions)| {
                (
                    PyBytes::new(py, &data).unbind(),
                    PyBytes::new(py, &positions).unbind(),
                )
            })
            .collect())
    }

    fn encode_frame(
        &self,
        py: Python<'_>,
        raw: &Bound<'_, PyBytes>,
        block_tokens: usize,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
        if block_tokens == 0 || block_tokens.checked_mul(4).is_none() {
            return Err(exceptions::PyValueError::new_err(
                "invalid token block size",
            ));
        }
        let raw = raw.as_bytes();
        let (data, entries) = py
            .detach(|| self.codec.encode_frame(raw, block_tokens))
            .ok_or_else(|| exceptions::PyValueError::new_err("invalid uint32 token block"))?;
        Ok((
            PyBytes::new(py, &data).unbind(),
            PyBytes::new(py, &entries).unbind(),
        ))
    }
}

#[pymethods]
impl TokenStorageDecoder {
    #[new]
    fn new(width: u8) -> PyResult<Self> {
        Ok(Self {
            width: storage_width(width)?,
        })
    }

    fn read(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyBytes>,
        bit_offset: usize,
        count: usize,
    ) -> PyResult<Py<PyAny>> {
        let data = data.as_bytes();
        let ids = py
            .detach(|| rmcp_token_storage::decode(data, self.width, bit_offset, count))
            .ok_or_else(|| {
                exceptions::PyValueError::new_err("packed token range exceeds stored bytes")
            })?;
        TiktokenBuffer::new(ids).into_py_any(py)
    }
}

fn source_io_error(py: Python<'_>, error: std::io::Error, path: &str) -> PyErr {
    if let Some(errno) = error.raw_os_error() {
        let message = py
            .import("os")
            .and_then(|os| os.call_method1("strerror", (errno,)))
            .and_then(|value| value.extract::<String>());
        match message {
            Ok(message) => exceptions::PyOSError::new_err((errno, message, path.to_owned())),
            Err(error) => error,
        }
    } else {
        exceptions::PyOSError::new_err(error.to_string())
    }
}

#[pyclass]
struct OwnedSourceCursor {
    cursor: SourceCursor,
    path: String,
}

#[pyclass(frozen)]
struct OwnedSourceBatch {
    batch: SourceBatch,
}

#[pyclass]
struct OwnedEncodedSourceBatch {
    provenance: Vec<Option<String>>,
    encoded: Option<Result<(Vec<Rank>, Vec<usize>), crate::EncodeError>>,
}

#[pymethods]
impl OwnedEncodedSourceBatch {
    #[getter]
    fn provenance(&self) -> Vec<Option<String>> {
        self.provenance.clone()
    }

    fn encoded(&mut self, py: Python<'_>) -> PyResult<(Py<PyAny>, Vec<usize>)> {
        // This is a single-use handoff to the Python encoding boundary. Keep
        // encoding failures here rather than turning them into source errors.
        let (ids, lengths) = self
            .encoded
            .take()
            .ok_or_else(|| exceptions::PyRuntimeError::new_err("source batch already consumed"))?
            .map_err(|error| exceptions::PyValueError::new_err(error.message))?;
        Ok((TiktokenBuffer::new(ids).into_py_any(py)?, lengths))
    }
}

impl CoreBPE {
    fn encode_source_rows(
        &self,
        batch: &SourceBatch,
        allowed: &HashSet<&str>,
        eos: Rank,
    ) -> Result<(Vec<Rank>, Vec<usize>), crate::EncodeError> {
        let rows = self.encode_batch_serial(batch.texts.iter().map(String::as_str), allowed)?;
        let lengths: Vec<usize> = rows.iter().map(|row| row.len() + 1).collect();
        let mut ids = Vec::with_capacity(lengths.iter().sum());
        for row in rows {
            ids.extend(row);
            ids.push(eos);
        }
        Ok((ids, lengths))
    }
}

#[pymethods]
impl OwnedSourceCursor {
    fn close(&mut self) {
        self.cursor.close();
    }

    fn read_batch(
        &mut self,
        py: Python<'_>,
        provenance: bool,
    ) -> PyResult<Option<OwnedSourceBatch>> {
        match py.detach(|| self.cursor.batch(provenance)) {
            Ok(batch) => Ok(batch.map(|batch| OwnedSourceBatch { batch })),
            Err(ReadError::Io(error)) => Err(source_io_error(py, error, &self.path)),
            Err(ReadError::Declined) => Err(exceptions::PyNotImplementedError::new_err(
                "source batch requires the original Python decoder before encoding",
            )),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyBytes>>> {
        match py.detach(|| self.cursor.raw_line()) {
            Ok(line) => Ok(line.map(|line| PyBytes::new(py, &line).unbind())),
            Err(error) => Err(source_io_error(py, error, &self.path)),
        }
    }
}

#[pymethods]
impl OwnedSourceBatch {
    #[getter]
    fn provenance(&self) -> Vec<Option<String>> {
        self.batch.provenance.clone()
    }
}

#[pymethods]
impl CoreBPE {
    fn _read_encode_source_batch(
        &self,
        py: Python<'_>,
        mut cursor: PyRefMut<'_, OwnedSourceCursor>,
        allowed_special: HashSet<PyBackedStr>,
        eos: Rank,
        provenance: bool,
    ) -> PyResult<Option<OwnedEncodedSourceBatch>> {
        let inner = &mut cursor.cursor;
        let result = py.detach(|| {
            let Some(batch) = inner.batch(provenance)? else {
                return Ok(None);
            };
            let allowed = allowed_special.iter().map(|s| s.as_ref()).collect();
            let encoded = self.encode_source_rows(&batch, &allowed, eos);
            Ok(Some(OwnedEncodedSourceBatch {
                provenance: batch.provenance,
                encoded: Some(encoded),
            }))
        });
        match result {
            Ok(batch) => Ok(batch),
            Err(ReadError::Io(error)) => Err(source_io_error(py, error, &cursor.path)),
            Err(ReadError::Declined) => Err(exceptions::PyNotImplementedError::new_err(
                "source batch requires the original Python decoder before encoding",
            )),
        }
    }

    #[pyo3(signature = (path, split, start, stop=None))]
    fn _open_source(
        &self,
        py: Python<'_>,
        path: String,
        split: String,
        start: u64,
        stop: Option<u64>,
    ) -> PyResult<OwnedSourceCursor> {
        let opened =
            py.detach(|| SourceCursor::open(std::path::Path::new(&path), split, start, stop));
        match opened {
            Ok(cursor) => Ok(OwnedSourceCursor { cursor, path }),
            Err(error) => Err(source_io_error(py, error, &path)),
        }
    }

    fn _encode_source_batch(
        &self,
        py: Python<'_>,
        batch: &OwnedSourceBatch,
        allowed_special: HashSet<PyBackedStr>,
        eos: Rank,
    ) -> PyResult<(Py<PyAny>, Vec<usize>)> {
        let result = py.detach(|| {
            let allowed: HashSet<&str> = allowed_special.iter().map(|s| s.as_ref()).collect();
            self.encode_source_rows(&batch.batch, &allowed, eos)
        });
        let (ids, lengths) =
            result.map_err(|error| exceptions::PyValueError::new_err(error.message))?;
        Ok((TiktokenBuffer::new(ids).into_py_any(py)?, lengths))
    }

    #[new]
    fn py_new(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> PyResult<Self> {
        Self::new_internal(encoder, special_tokens_encoder, pattern)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))
    }

    #[staticmethod]
    fn _from_packed_ranks(
        ranks: &Bound<'_, PyBytes>,
        special_tokens: HashMap<String, Rank>,
        pattern: &str,
    ) -> PyResult<(Self, Rank)> {
        let encoder = packed_ranks(ranks.as_bytes())?;
        let maximum = encoder
            .values()
            .chain(special_tokens.values())
            .copied()
            .max()
            .unwrap();
        let core = Self::new_internal(encoder, special_tokens, pattern)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))?;
        Ok((core, maximum))
    }

    fn _mergeable_ranks(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let ranks = pyo3::types::PyDict::new(py);
        for (rank, word) in &self.decoder {
            ranks.set_item(PyBytes::new(py, word), rank)?;
        }
        Ok(ranks.into_any().unbind())
    }

    // ====================
    // Encoding
    // ====================

    #[pyo3(name = "encode_ordinary")]
    fn py_encode_ordinary(&self, py: Python, text: &str) -> Vec<Rank> {
        py.detach(|| self.encode_ordinary(text))
    }

    #[pyo3(name = "encode")]
    fn py_encode(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Vec<Rank>> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            match self.encode(text, &allowed_special) {
                Ok((tokens, _)) => Ok(tokens),
                Err(e) => Err(PyErr::new::<exceptions::PyValueError, _>(e.message)),
            }
        })
    }

    /// Summarize the original IDs without materializing their Python list.
    #[pyo3(name = "_causal_token_summary")]
    fn py_causal_token_summary(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
        eos: Rank,
    ) -> PyResult<(usize, usize, bool)> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            let (tokens, _) = self
                .encode(text, &allowed_special)
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.message))?;
            Ok((
                tokens.len(),
                tokens.iter().filter(|token| **token == eos).count(),
                tokens.last() == Some(&eos),
            ))
        })
    }

    #[pyo3(name = "_encode_with_character_offsets")]
    fn py_encode_with_character_offsets(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<(Vec<Rank>, Vec<(usize, usize)>)> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self.encode_with_character_offsets(text, &allowed_special)
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.message))
        })
    }

    #[pyo3(name = "_encode_with_character_offsets_prefixes")]
    fn py_encode_with_character_offsets_prefixes(
        &self,
        py: Python,
        text: &str,
        ends: Vec<usize>,
        allowed_special: HashSet<PyBackedStr>,
        max_tokens: usize,
    ) -> PyResult<Py<PyList>> {
        let rows = py.detach(|| {
            let allowed: HashSet<&str> = allowed_special.iter().map(|s| s.as_ref()).collect();
            self.encode_with_character_offsets_prefixes(text, &ends, &allowed, max_tokens)
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.message))
        })?;
        let result = PyList::empty(py);
        let mut previous: Option<(Bound<'_, PyList>, Bound<'_, PyList>)> = None;
        for (index, (ids, offsets)) in rows.iter().enumerate() {
            let common = index.checked_sub(1).map_or(0, |prior| {
                ids.iter()
                    .zip(offsets)
                    .zip(rows[prior].0.iter().zip(&rows[prior].1))
                    .take_while(|(left, right)| left == right)
                    .count()
            });
            let (py_ids, py_offsets) = if let Some((prior_ids, prior_offsets)) =
                previous.as_ref().filter(|_| common > 0)
            {
                // Slices own independent lists; only immutable integer/tuple values are shared.
                let py_ids = prior_ids.get_slice(0, common);
                let py_offsets = prior_offsets.get_slice(0, common);
                for id in &ids[common..] {
                    py_ids.append(*id)?;
                }
                for offset in &offsets[common..] {
                    py_offsets.append(*offset)?;
                }
                (py_ids, py_offsets)
            } else {
                (
                    PyList::new(py, ids.iter().copied())?,
                    PyList::new(py, offsets.iter().copied())?,
                )
            };
            result.append((&py_ids, &py_offsets))?;
            previous = Some((py_ids, py_offsets));
        }
        Ok(result.unbind())
    }

    /// Project fresh prefix offsets to the target bounds consumed by causal rows.
    /// The consumer retains its existing ordered length/nonempty-target checks.
    #[pyo3(name = "_encode_with_character_targets_prefixes")]
    #[pyo3(signature = (text, spans, allowed_special, max_tokens, *, as_json=false))]
    fn py_encode_with_character_targets_prefixes(
        &self,
        py: Python,
        text: &str,
        spans: Vec<(usize, usize)>,
        allowed_special: HashSet<PyBackedStr>,
        max_tokens: usize,
        as_json: bool,
    ) -> PyResult<Py<PyList>> {
        let rows = py.detach(|| {
            let allowed: HashSet<&str> = allowed_special.iter().map(|s| s.as_ref()).collect();
            self.encode_with_character_targets_prefixes(text, &spans, &allowed, max_tokens)
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.message))
        })?;
        let result = PyList::empty(py);
        let mut previous: Option<Bound<'_, PyList>> = None;
        for (index, (ids, first, last)) in rows.iter().enumerate() {
            if as_json {
                let payload = py
                    .detach(|| serde_json::to_vec(ids))
                    .map_err(|error| exceptions::PyValueError::new_err(error.to_string()))?;
                result.append((
                    PyBytes::new(py, &payload),
                    ids.len(),
                    *first,
                    *last,
                    ids.iter().copied().max(),
                ))?;
                continue;
            }
            let common = index.checked_sub(1).map_or(0, |prior| {
                ids.iter()
                    .zip(&rows[prior].0)
                    .take_while(|(left, right)| left == right)
                    .count()
            });
            let py_ids = if let Some(prior) = previous.as_ref().filter(|_| common > 0) {
                let values = prior.get_slice(0, common);
                for id in &ids[common..] {
                    values.append(*id)?;
                }
                values
            } else {
                PyList::new(py, ids.iter().copied())?
            };
            result.append((&py_ids, *first, *last))?;
            previous = Some(py_ids);
        }
        Ok(result.unbind())
    }

    #[pyo3(name = "count_tokens")]
    fn py_count_tokens(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<usize> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self.count_tokens(text, &allowed_special)
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.message))
        })
    }

    /// Count UTF-8 byte ranges without extracting a Python string per range.
    #[pyo3(name = "count_tokens_ranges")]
    fn py_count_tokens_ranges(
        &self,
        py: Python,
        text: &str,
        ranges: Vec<(usize, usize)>,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Vec<usize>> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            if let Some(counts) = self.try_count_ranges_indexed(text, &ranges, &allowed_special) {
                return Ok(counts);
            }
            ranges
                .into_iter()
                .map(|(start, end)| {
                    let span = text.get(start..end).ok_or_else(|| {
                        PyErr::new::<exceptions::PyValueError, _>(
                            "range must be ordered, in bounds and on UTF-8 boundaries",
                        )
                    })?;
                    self.count_tokens(span, &allowed_special)
                        .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.message))
                })
                .collect()
        })
    }

    fn encode_to_tiktoken_buffer(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Py<PyAny>> {
        let tokens_res = py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self.encode(text, &allowed_special)
        });

        let tokens = match tokens_res {
            Ok((tokens, _)) => tokens,
            Err(e) => return Err(PyErr::new::<exceptions::PyValueError, _>(e.message)),
        };

        let buffer = TiktokenBuffer::new(tokens);
        buffer.into_py_any(py)
    }

    fn encode_to_tiktoken_buffers(
        &self,
        py: Python,
        texts: Vec<PyBackedStr>,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let rows = py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self.encode_batch_serial(texts.iter().map(|text| text.as_ref()), &allowed_special)
        });
        let rows = match rows {
            Ok(rows) => rows,
            Err(e) => return Err(PyErr::new::<exceptions::PyValueError, _>(e.message)),
        };
        rows.into_iter()
            .map(|tokens| TiktokenBuffer::new(tokens).into_py_any(py))
            .collect()
    }

    fn _encode_bytes(&self, py: Python, bytes: &[u8]) -> Vec<Rank> {
        py.detach(|| {
            match std::str::from_utf8(bytes) {
                // Straightforward case
                Ok(text) => self.encode_ordinary(text),
                // Oops, don't actually have UTF-8. But we need to do the regex splitting in
                // Unicode space, so we make our best guess at where we would have splits
                Err(e) => {
                    let text = unsafe { std::str::from_utf8_unchecked(&bytes[..e.valid_up_to()]) };
                    let (tokens, last_piece_token_len) =
                        self.encode(text, &HashSet::new()).unwrap();
                    let (mut tokens, last_piece_token_len) =
                        self._increase_last_piece_token_len(tokens, last_piece_token_len);

                    let mut unstable_bytes;
                    if !tokens.is_empty() && last_piece_token_len > 0 {
                        // Lop off the tokens from the last piece and run BPE on the remaining bytes
                        // This likely matches what models see better, e.g. if you assume we're
                        // dealing with truncated UTF-8 bytes.
                        // Niche, but note this may not be correct if we'd have had a regex
                        // split between the valid UTF-8 and the invalid bytes.
                        unstable_bytes = self
                            .decode_bytes(&tokens[tokens.len() - last_piece_token_len..])
                            .unwrap();
                        unstable_bytes.extend_from_slice(&bytes[e.valid_up_to()..]);

                        tokens.truncate(tokens.len() - last_piece_token_len);
                    } else {
                        unstable_bytes = bytes[e.valid_up_to()..].to_vec();
                    }

                    if !unstable_bytes.is_empty() {
                        match self.encoder.get(&unstable_bytes) {
                            Some(token) => tokens.push(*token),
                            None => {
                                tokens.extend(&_byte_pair_encode(&unstable_bytes, &self.encoder))
                            }
                        }
                    }
                    tokens
                }
            }
        })
    }

    #[pyo3(name = "encode_with_unstable")]
    fn py_encode_with_unstable(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<(Vec<Rank>, Py<PyList>)> {
        let (tokens, completions): (Vec<Rank>, HashSet<Vec<Rank>>) = py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            self._encode_unstable_native(text, &allowed_special)
        });
        let py_completions = PyList::new(py, completions.into_iter())?;
        Ok((tokens, py_completions.into()))
    }

    fn encode_single_token(&self, piece: &[u8]) -> PyResult<Rank> {
        if let Some(token) = self.encoder.get(piece).copied() {
            return Ok(token);
        }
        if let Ok(piece_str) = std::str::from_utf8(piece) {
            if let Some(token) = self.special_tokens_encoder.get(piece_str).copied() {
                return Ok(token);
            }
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(piece.to_owned()))
    }

    fn encode_single_piece(&self, piece: &[u8]) -> Vec<Rank> {
        if let Some(token) = self.encoder.get(piece) {
            return vec![*token];
        }
        _byte_pair_encode(piece, &self.encoder)
    }

    // ====================
    // Decoding
    // ====================

    #[pyo3(name = "decode_bytes")]
    fn py_decode_bytes(&self, py: Python, tokens: Vec<Rank>) -> Result<Py<PyBytes>, PyErr> {
        match py.detach(|| self.decode_bytes(&tokens)) {
            Ok(bytes) => Ok(PyBytes::new(py, &bytes).into()),
            Err(e) => Err(pyo3::exceptions::PyKeyError::new_err(format!("{}", e))),
        }
    }

    fn decode_single_token_bytes(&self, py: Python, token: Rank) -> PyResult<Py<PyBytes>> {
        if let Some(bytes) = self.decoder.get(&token) {
            return Ok(PyBytes::new(py, bytes).into());
        }
        if let Some(bytes) = self.special_tokens_decoder.get(&token) {
            return Ok(PyBytes::new(py, bytes).into());
        }
        Err(PyErr::new::<exceptions::PyKeyError, _>(token.to_string()))
    }

    // ====================
    // Miscellaneous
    // ====================

    fn token_byte_values(&self, py: Python) -> Vec<Py<PyBytes>> {
        self.sorted_token_bytes
            .iter()
            .map(|x| PyBytes::new(py, x).into())
            .collect()
    }
}

#[pyclass(frozen)]
struct TiktokenBuffer {
    tokens: Vec<Rank>,
    length: isize,
}

impl TiktokenBuffer {
    fn new(tokens: Vec<Rank>) -> Self {
        Self {
            length: tokens.len() as isize,
            tokens,
        }
    }
}

static RANK_ITEMSIZE: isize = std::mem::size_of::<Rank>() as isize;

#[pymethods]
impl TiktokenBuffer {
    // Based on https://github.com/PyO3/pyo3/blob/v0.22.2/tests/test_buffer_protocol.rs#L25
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Object is not writable",
            ));
        }
        unsafe {
            let view_ref = &mut *view;
            view_ref.obj = slf.clone().into_any().into_ptr();

            let data = slf.get();
            view_ref.buf = data.tokens.as_ptr() as *mut std::os::raw::c_void;
            view_ref.len = data.length * RANK_ITEMSIZE;
            view_ref.readonly = 1;
            view_ref.itemsize = RANK_ITEMSIZE;
            view_ref.format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
                c"I".as_ptr().cast_mut()
            } else {
                std::ptr::null_mut()
            };
            view_ref.ndim = 1;
            view_ref.shape = if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
                // The owner is frozen and retained by view_ref.obj. Shape counts
                // elements, not bytes, and must survive copies of Py_buffer.
                std::ptr::from_ref(&data.length).cast_mut()
            } else {
                std::ptr::null_mut()
            };
            view_ref.strides = if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
                std::ptr::from_ref(&RANK_ITEMSIZE).cast_mut()
            } else {
                std::ptr::null_mut()
            };
            view_ref.suboffsets = std::ptr::null_mut();
            view_ref.internal = std::ptr::null_mut();
        }

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut pyo3::ffi::Py_buffer) {
        // Metadata is static or retained by the frozen owner. Keep this slot:
        // NumPy uses its presence to retain the original memoryview-based owner.
    }
}

/// Cache UTF-8 views for a complete batch before any encoder work begins.
#[pyfunction]
fn _prepare_utf8(_texts: Vec<PyBackedStr>) {}

#[pymodule(gil_used = false)]
fn _tiktoken(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        crate::rmcp_model::_bounded_model_vocab,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::rmcp_model::_prove_model_words, m)?)?;
    m.add_function(wrap_pyfunction!(crate::rmcp_model::_model_word_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(crate::rmcp_model::_admit_model_words, m)?)?;

    m.add_class::<CoreBPE>()?;
    m.add_class::<TokenStorageEncoder>()?;
    m.add_class::<TokenStorageDecoder>()?;
    m.add_class::<TokenTierCodec>()?;
    m.add_function(wrap_pyfunction!(_prepare_utf8, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::rmcp_token_bits::_unpack_bits17_into,
        m
    )?)?;
    Ok(())
}
