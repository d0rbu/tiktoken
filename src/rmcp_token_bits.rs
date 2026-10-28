//! Decode RMCP's little-endian 17-bit storage into an existing token tensor.

use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyBufferError, PyValueError},
    prelude::*,
};

#[pyfunction]
pub fn _unpack_bits17_into(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    start_bit: usize,
    destination: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let source = PyBuffer::<u8>::get(source)?;
    let input = source
        .as_slice(py)
        .ok_or_else(|| PyValueError::new_err("packed token bytes must be contiguous"))?;
    let destination = PyBuffer::<i64>::get(destination)?;
    // PyO3 0.29.3 admits a '>' buffer on little-endian hosts. It would write
    // native bytes with the wrong numerical meaning; the caller can fall back.
    let prefix = destination.format().to_bytes().first().copied();
    let opposite_endian = if cfg!(target_endian = "little") {
        matches!(prefix, Some(b'>' | b'!'))
    } else {
        prefix == Some(b'<')
    };
    if opposite_endian {
        return Err(PyBufferError::new_err(
            "decoded destination must use native byte order",
        ));
    }
    let output = destination.as_mut_slice(py).ok_or_else(|| {
        PyValueError::new_err("decoded destination must be writable and contiguous")
    })?;
    let end = output
        .len()
        .checked_mul(17)
        .and_then(|bits| start_bit.checked_add(bits));
    if end.is_none_or(|end| end > input.len().saturating_mul(8)) {
        return Err(PyValueError::new_err(
            "packed token window exceeds its byte buffer",
        ));
    }
    // Readonly exports can still alias a writable destination. The caller must
    // retain its existing path for these: a streaming decode changes its reads.
    let input_range = input.as_ptr_range();
    let output_range = output.as_ptr_range();
    if !output.is_empty()
        && (input_range.start as usize) < (output_range.end as usize)
        && (output_range.start as usize) < (input_range.end as usize)
    {
        return Err(PyBufferError::new_err(
            "packed source and decoded destination must not overlap",
        ));
    }
    for (index, target) in output.iter().enumerate() {
        let bit = start_bit + index * 17;
        let byte = bit / 8;
        let word = input[byte].get() as u32
            | (input[byte + 1].get() as u32) << 8
            | (input[byte + 2].get() as u32) << 16;
        target.set(((word >> (bit % 8)) & 0x1ffff) as i64);
    }
    Ok(())
}
