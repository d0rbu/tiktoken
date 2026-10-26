//! CPU storage conversion of already admitted token IDs, independent of BPE.

pub(crate) struct Encoder {
    width: u8,
    pending: [u32; 8],
    length: usize,
    output: Vec<u8>,
}

impl Encoder {
    pub(crate) fn new(width: u8) -> Self {
        Self {
            width,
            pending: [0; 8],
            length: 0,
            output: Vec::new(),
        }
    }

    /// Input is the producer's complete little-endian uint32 frame.
    pub(crate) fn push(&mut self, raw: &[u8]) -> &[u8] {
        self.output.clear();
        self.output
            .reserve((self.length + raw.len() / 4) * usize::from(self.width) / 8 + 32);
        let mut words = raw
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        if self.length != 0 {
            while self.length < 8 {
                match words.next() {
                    Some(id) => {
                        self.pending[self.length] = id;
                        self.length += 1;
                    }
                    None => return &self.output,
                }
            }
            block(self.width, &self.pending, &mut self.output);
            self.length = 0;
        }
        loop {
            for i in 0..8 {
                match words.next() {
                    Some(id) => self.pending[i] = id,
                    None => {
                        self.length = i;
                        return &self.output;
                    }
                }
            }
            block(self.width, &self.pending, &mut self.output);
        }
    }

    /// Append an already packed range without expanding its complete ID array.
    pub(crate) fn push_packed(
        &mut self,
        data: &[u8],
        position: usize,
        count: usize,
    ) -> Option<&[u8]> {
        let width = usize::from(self.width);
        let bits = count.checked_mul(width)?;
        if position.checked_add(bits)? > data.len().checked_mul(8)? {
            return None;
        }
        self.output.clear();
        self.output.reserve(bits / 8 + self.length * width / 8 + 32);
        let mut at = position;
        let mut remaining = count;
        if self.length != 0 {
            while self.length < 8 && remaining != 0 {
                self.pending[self.length] = one(data, at, self.width);
                self.length += 1;
                at += width;
                remaining -= 1;
            }
            if self.length < 8 {
                return Some(&self.output);
            }
            block(self.width, &self.pending, &mut self.output);
            self.length = 0;
        }
        let bytes = remaining / 8 * width;
        copy_packed_bytes(data, at, bytes, &mut self.output);
        at += bytes * 8;
        remaining %= 8;
        for i in 0..remaining {
            self.pending[i] = one(data, at, self.width);
            at += width;
        }
        self.length = remaining;
        Some(&self.output)
    }

    /// Emit only a successful end's remaining IDs and zero high padding bits.
    pub(crate) fn finish(&mut self) -> &[u8] {
        self.output.clear();
        let mut word = 0u64;
        let mut bits = 0;
        for &id in &self.pending[..self.length] {
            word |= u64::from(id) << bits;
            bits += u32::from(self.width);
            while bits >= 8 {
                self.output.push(word as u8);
                word >>= 8;
                bits -= 8;
            }
        }
        if bits != 0 {
            self.output.push(word as u8);
        }
        self.length = 0;
        &self.output
    }
}

/// The admitted range contains complete groups of eight IDs (whole bytes).
fn copy_packed_bytes(data: &[u8], position: usize, count: usize, output: &mut Vec<u8>) {
    let begin = position / 8;
    let shift = position % 8;
    if shift == 0 {
        output.extend_from_slice(&data[begin..begin + count]);
        return;
    }
    let words = count / 8 * 8;
    for offset in (0..words).step_by(8) {
        let at = begin + offset;
        let low = u64::from_le_bytes(data[at..at + 8].try_into().unwrap());
        let next = u64::from(data[at + 8]);
        output.extend_from_slice(&((low >> shift) | (next << (64 - shift))).to_le_bytes());
    }
    for offset in words..count {
        let at = begin + offset;
        output.push((data[at] >> shift) | (data[at + 1] << (8 - shift)));
    }
}

#[inline]
fn block(width: u8, ids: &[u32; 8], output: &mut Vec<u8>) {
    if width == 17 {
        let a = u64::from(ids[0])
            | u64::from(ids[1]) << 17
            | u64::from(ids[2]) << 34
            | u64::from(ids[3]) << 51;
        let b = u64::from(ids[3]) >> 13
            | u64::from(ids[4]) << 4
            | u64::from(ids[5]) << 21
            | u64::from(ids[6]) << 38
            | u64::from(ids[7]) << 55;
        output.extend_from_slice(&a.to_le_bytes());
        output.extend_from_slice(&b.to_le_bytes());
        output.push((ids[7] >> 9) as u8);
        return;
    }
    let mut words = [0u64; 4];
    for (i, &id) in ids.iter().enumerate() {
        let position = i * usize::from(width);
        let word = position / 64;
        let shift = position % 64;
        words[word] |= u64::from(id) << shift;
        if shift + usize::from(width) > 64 {
            words[word + 1] |= u64::from(id) >> (64 - shift);
        }
    }
    let mut bytes = [0; 32];
    for (i, word) in words.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&word.to_le_bytes());
    }
    output.extend_from_slice(&bytes[..usize::from(width)]);
}

fn one(data: &[u8], position: usize, width: u8) -> u32 {
    let at = position / 8;
    let end = data.len().min(at + 5);
    let mut bytes = [0; 8];
    bytes[..end - at].copy_from_slice(&data[at..end]);
    ((u64::from_le_bytes(bytes) >> (position % 8)) & ((1u64 << width) - 1)) as u32
}

/// Check a consumed external byte range once, then decode into an owned buffer.
pub(crate) fn decode(data: &[u8], width: u8, position: usize, count: usize) -> Option<Vec<u32>> {
    if position.checked_add(count.checked_mul(usize::from(width))?)? > data.len().checked_mul(8)? {
        return None;
    }
    let mut result = Vec::with_capacity(count);
    let mut at = position;
    while result.len() < count && at % 8 != 0 {
        result.push(one(data, at, width));
        at += usize::from(width);
    }
    if width == 17 {
        while count - result.len() >= 8 {
            let begin = at / 8;
            let a = u64::from_le_bytes(data[begin..begin + 8].try_into().unwrap());
            let b = u64::from_le_bytes(data[begin + 8..begin + 16].try_into().unwrap());
            let mask = (1u64 << 17) - 1;
            result.extend_from_slice(&[
                (a & mask) as u32,
                ((a >> 17) & mask) as u32,
                ((a >> 34) & mask) as u32,
                (((a >> 51) | (b << 13)) & mask) as u32,
                ((b >> 4) & mask) as u32,
                ((b >> 21) & mask) as u32,
                ((b >> 38) & mask) as u32,
                ((b >> 55) | (u64::from(data[begin + 16]) << 9)) as u32,
            ]);
            at += 136;
        }
    }
    while result.len() < count {
        result.push(one(data, at, width));
        at += usize::from(width);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle(ids: &[u32], width: u8) -> Vec<u8> {
        let width = usize::from(width);
        let mut bytes = vec![0; (ids.len() * width + 7) / 8];
        for (i, &id) in ids.iter().enumerate() {
            for bit in 0..width {
                bytes[(i * width + bit) / 8] |=
                    (((id >> bit) & 1) as u8) << ((i * width + bit) % 8);
            }
        }
        bytes
    }

    fn shifted(ids: &[u32], width: u8, offset: usize) -> Vec<u8> {
        let mut bytes = vec![0xa5; (offset + ids.len() * usize::from(width)).div_ceil(8) + 2];
        for (i, &id) in ids.iter().enumerate() {
            for bit in 0..usize::from(width) {
                let at = offset + i * usize::from(width) + bit;
                bytes[at / 8] &= !(1 << (at % 8));
                bytes[at / 8] |= (((id >> bit) & 1) as u8) << (at % 8);
            }
        }
        bytes
    }

    #[test]
    fn mixed_raw_packed_edges_all_widths_offsets_and_pending_lengths_match_bits() {
        for width in 8..=32 {
            let mask = ((1u64 << width) - 1) as u32;
            let all: Vec<u32> = (0u32..280)
                .map(|i| match i % 3 {
                    0 => 0,
                    1 => mask,
                    _ => i.wrapping_mul(0x9e3779b9) & mask,
                })
                .collect();
            for prefix in 0..8 {
                for count in [0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 257] {
                    for offset in 0..8 {
                        let mut encoder = Encoder::new(width);
                        let mut actual = Vec::new();
                        let left: Vec<u8> =
                            all[..prefix].iter().flat_map(|x| x.to_le_bytes()).collect();
                        actual.extend_from_slice(encoder.push(&left));
                        let packed = shifted(&all[prefix..prefix + count], width, offset);
                        actual.extend_from_slice(
                            encoder.push_packed(&packed, offset, count).unwrap(),
                        );
                        let tail = &all[prefix + count..prefix + count + 9];
                        let raw: Vec<u8> = tail.iter().flat_map(|x| x.to_le_bytes()).collect();
                        actual.extend_from_slice(encoder.push(&raw));
                        actual.extend_from_slice(encoder.finish());
                        assert_eq!(actual, oracle(&all[..prefix + count + 9], width));
                        assert!(encoder.finish().is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn successive_packed_ranges_keep_arbitrary_frame_and_logical_shard_boundaries() {
        for width in 8..=32 {
            let mask = ((1u64 << width) - 1) as u32;
            let ids: Vec<u32> = (0u32..257)
                .map(|i| i.wrapping_mul(0x71ad4be5) & mask)
                .collect();
            for offset in 0..8 {
                let data = shifted(&ids, width, offset);
                for step in [1, 7, 8, 9, 33, 64] {
                    let mut encoder = Encoder::new(width);
                    let mut actual = Vec::new();
                    for first in (0..ids.len()).step_by(step) {
                        let count = step.min(ids.len() - first);
                        actual.extend_from_slice(
                            encoder
                                .push_packed(&data, offset + first * usize::from(width), count)
                                .unwrap(),
                        );
                    }
                    actual.extend_from_slice(encoder.finish());
                    assert_eq!(actual, oracle(&ids, width));
                }
                for cut in [1, 7, 8, 9, 31, 128, 256] {
                    let mut encoder = Encoder::new(width);
                    let mut left = encoder.push_packed(&data, offset, cut).unwrap().to_vec();
                    left.extend_from_slice(encoder.finish());
                    let mut right = encoder
                        .push_packed(&data, offset + cut * usize::from(width), ids.len() - cut)
                        .unwrap()
                        .to_vec();
                    right.extend_from_slice(encoder.finish());
                    assert_eq!(left, oracle(&ids[..cut], width));
                    assert_eq!(right, oracle(&ids[cut..], width));
                }
            }
        }
    }

    #[test]
    fn invalid_packed_ranges_leave_pending_state_unchanged() {
        for (data, position, count) in [
            (&[][..], 0, 1),
            (&[0][..], 0, 1),
            (&[0; 17][..], 1, 8),
            (&[0; 17][..], 0, 9),
            (&[0; 17][..], usize::MAX, 1),
            (&[0; 17][..], 0, usize::MAX),
            (&[0][..], 9, 0),
        ] {
            let mut encoder = Encoder::new(17);
            let ids: Vec<u32> = (0..20).map(|i| i * 3001).collect();
            let prefix: Vec<u8> = ids[..13].iter().flat_map(|x| x.to_le_bytes()).collect();
            let mut actual = encoder.push(&prefix).to_vec();
            assert!(encoder.push_packed(data, position, count).is_none());
            assert_eq!(encoder.output, actual);
            assert_eq!(encoder.length, 5);
            let suffix: Vec<u8> = ids[13..].iter().flat_map(|x| x.to_le_bytes()).collect();
            actual.extend_from_slice(encoder.push(&suffix));
            actual.extend_from_slice(encoder.finish());
            assert_eq!(actual, oracle(&ids, 17));
        }
    }

    #[test]
    fn all_widths_segmented_frames_and_ranges_match_independent_bits() {
        let mut random = 0x71ad4be51c024d8bu64;
        for width in 8..=32 {
            let mask = ((1u64 << width) - 1) as u32;
            for length in [0, 1, 2, 7, 8, 9, 15, 16, 17, 257] {
                let ids: Vec<u32> = (0..length)
                    .map(|i| {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        match i % 3 {
                            0 => 0,
                            1 => mask,
                            _ => random as u32 & mask,
                        }
                    })
                    .collect();
                let raw: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
                for step in [1, 2, 7, 8, 9, 33] {
                    let mut encoder = Encoder::new(width);
                    let mut all = Vec::new();
                    for frame in raw.chunks(step * 4) {
                        all.extend_from_slice(encoder.push(frame));
                    }
                    all.extend_from_slice(encoder.finish());
                    assert_eq!(all, oracle(&ids, width));
                    assert!(encoder.finish().is_empty());
                    for start in 0..=length {
                        for wanted in [0, 1, 7, 8, 9, 35] {
                            let end = (start + wanted).min(length);
                            assert_eq!(
                                decode(&all, width, start * usize::from(width), end - start),
                                Some(ids[start..end].to_vec())
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn consumed_short_and_overflowed_ranges_are_rejected() {
        assert_eq!(decode(&[], 17, 0, 0), Some(vec![]));
        assert_eq!(decode(&[], 17, 0, 1), None);
        assert_eq!(decode(&[0; 17], 17, 0, 9), None);
        assert_eq!(decode(&[0; 17], 17, 1, 8), None);
        assert_eq!(decode(&[0; 17], 17, usize::MAX, 1), None);
        assert_eq!(decode(&[0; 17], 17, 0, usize::MAX), None);
    }
}
