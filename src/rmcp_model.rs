//! Compiled versions of the existing private model-admission loops.

use pyo3::{
    exceptions::PyKeyError,
    prelude::*,
    pyclass::CompareOp,
    types::{PyBytes, PyDict, PyInt, PyString},
};
use rustc_hash::FxHashSet;

/// Preserve the Python admission order while keeping the per-merge loop native.
#[pyfunction]
pub(crate) fn _admit_model_words<'py>(
    py: Python<'py>,
    raw_vocab: &Bound<'py, PyDict>,
    merges: &Bound<'py, PyAny>,
    alphabet: &Bound<'py, PyDict>,
    max_word_chars: usize,
    max_total_chars: usize,
) -> PyResult<Option<(Bound<'py, PyDict>, Bound<'py, PyDict>)>> {
    let vocab = PyDict::new(py);
    let mut ids = FxHashSet::default();
    for (word, ordinal) in raw_vocab.iter() {
        if !ordinal.is_exact_instance_of::<PyInt>() {
            return Ok(None);
        }
        let Ok(id) = ordinal.extract::<u32>() else {
            return Ok(None);
        };
        vocab.set_item(word, ordinal)?;
        if !ids.insert(id) {
            return Ok(None);
        }
    }
    let words = PyDict::new(py);
    for (character, _) in alphabet.iter() {
        let Some(ordinal) = vocab.get_item(&character)? else {
            return Ok(None);
        };
        if ordinal.extract::<u32>()? == u32::MAX {
            return Ok(None);
        }
        words.set_item(character, ordinal)?;
    }
    let mut total = 256usize;
    let mut previous = None::<u32>;
    for pair in merges.try_iter()? {
        let pair = pair?;
        let mut pieces = pair.try_iter()?;
        let left = pieces.next().transpose()?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "not enough values to unpack (expected 2, got 0)",
            )
        })?;
        let right = pieces.next().transpose()?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "not enough values to unpack (expected 2, got 1)",
            )
        })?;
        if pieces.next().transpose()?.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "too many values to unpack (expected 2)",
            ));
        }
        let size = left.len()?.checked_add(right.len()?);
        let Some(size) = size else { return Ok(None) };
        let Some(next) = total.checked_add(size) else {
            return Ok(None);
        };
        total = next;
        if size > max_word_chars || total > max_total_chars {
            return Ok(None);
        }
        let Some(left_id) = words.get_item(&left)? else {
            return Ok(None);
        };
        let Some(right_id) = words.get_item(&right)? else {
            return Ok(None);
        };
        // Concatenation uses the same Python string operation as the original loop.
        let word = left.add(&right)?;
        let Some(output) = vocab.get_item(&word)? else {
            return Ok(None);
        };
        let output_id = output.extract::<u32>()?;
        if output_id == u32::MAX {
            return Ok(None);
        }
        if words.contains(&word)?
            || output_id
                <= previous
                    .unwrap_or(0)
                    .max(left_id.extract()?)
                    .max(right_id.extract()?)
        {
            return Ok(None);
        }
        words.set_item(word, output)?;
        previous = Some(output_id);
    }
    Ok(Some((vocab, words)))
}

#[pyfunction]
pub(crate) fn _bounded_model_vocab(
    vocab: &Bound<'_, PyDict>,
    max_entries: u32,
    max_word_chars: usize,
    max_total_bytes: usize,
) -> PyResult<bool> {
    let mut total = 0usize;
    let mut seen = FxHashSet::default();
    for (word, ordinal) in vocab.iter() {
        if !word.is_exact_instance_of::<PyString>()
            || word.len()? > max_word_chars
            || !ordinal.is_exact_instance_of::<PyInt>()
        {
            return Ok(false);
        }
        let Ok(id) = ordinal.extract::<u32>() else {
            return Ok(false);
        };
        if id == u32::MAX || id >= max_entries || !seen.insert(id) {
            return Ok(false);
        }
        let text = word.cast::<PyString>()?.to_str()?;
        let Some(next) = total.checked_add(text.len()) else {
            return Ok(false);
        };
        total = next;
        if total > max_total_bytes {
            return Ok(false);
        }
    }
    Ok(true)
}

#[pyfunction]
pub(crate) fn _prove_model_words(
    words: &Bound<'_, PyDict>,
    tokenize: &Bound<'_, PyAny>,
    mut remaining: usize,
) -> PyResult<bool> {
    for (word, expected) in words.iter() {
        let length = word.len()?;
        let cost = length.checked_mul(length.saturating_sub(1)).map(|x| x / 2);
        let Some(next) = cost.and_then(|cost| remaining.checked_sub(cost)) else {
            return Ok(false);
        };
        remaining = next;
        let tokens = tokenize.call1((&word,))?;
        if tokens.len()? != 1
            || tokens
                .get_item(0)?
                .getattr("id")?
                .rich_compare(&expected, CompareOp::Ne)?
                .is_truthy()?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

#[pyfunction]
pub(crate) fn _model_word_bytes<'py>(
    py: Python<'py>,
    words: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut alphabet = [u16::MAX; 324];
    let mut next = 256;
    for byte in 0..256 {
        if (33..127).contains(&byte) || (161..173).contains(&byte) || (174..256).contains(&byte) {
            alphabet[byte] = byte as u16;
        } else {
            alphabet[next] = byte as u16;
            next += 1;
        }
    }
    let output = PyDict::new(py);
    for (word, ordinal) in words.iter() {
        let text = word.cast::<PyString>()?.to_str()?;
        let mut bytes = Vec::with_capacity(text.len());
        for ch in text.chars() {
            let value = alphabet.get(ch as usize).copied().unwrap_or(u16::MAX);
            if value == u16::MAX {
                return Err(PyKeyError::new_err(ch.to_string()));
            }
            bytes.push(value as u8);
        }
        output.set_item(PyBytes::new(py, &bytes), ordinal)?;
    }
    Ok(output)
}
