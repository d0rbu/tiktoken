//! Experimental fixed-width common-token tiers; no persistent format selects it.

pub(crate) struct Codec {
    codes: Vec<(u32, u8)>,
    order: Vec<u32>,
    widths: [usize; 4],
    masks: [u32; 4],
    bases: [usize; 4],
}

pub(crate) struct IndexedRange<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) checkpoints: &'a [u8],
    pub(crate) tokens: usize,
    pub(crate) start: usize,
    pub(crate) count: usize,
}

impl Codec {
    pub(crate) fn new(order: &[u32], a: u8, b: u8) -> Option<Self> {
        if !(1..=8).contains(&a) || b <= a || b > 14 {
            return None;
        }
        let small = 1usize << a;
        if order.len() != small + (1usize << b) {
            return None;
        }
        let mut seen = vec![false; 1 << 17];
        let mut codes: Vec<_> = (0..1 << 17).map(|id| ((3u32 << 17) | id, 19)).collect();
        for (index, &id) in order.iter().enumerate() {
            if *seen.get(id as usize)? {
                return None;
            }
            seen[id as usize] = true;
            let (code, width) = if index < small {
                (index as u32, a + 1)
            } else {
                ((2u32 << b) | (index - small) as u32, b + 2)
            };
            codes[id as usize] = (code, width);
        }
        Some(Self {
            codes,
            order: order.to_vec(),
            widths: [
                usize::from(a + 1),
                usize::from(a + 1),
                usize::from(b + 2),
                19,
            ],
            masks: [(1 << a) - 1, (1 << a) - 1, (1 << b) - 1, (1 << 17) - 1],
            bases: [0, 0, small, 0],
        })
    }

    pub(crate) fn encode(&self, raw: &[u8]) -> Option<Vec<u8>> {
        self.encode_inner::<false>(raw).map(|(bytes, _)| bytes)
    }

    pub(crate) fn encode_indexed(&self, raw: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        self.encode_inner::<true>(raw)
    }

    pub(crate) fn encode_blocks(
        &self,
        raw: &[u8],
        block_tokens: usize,
    ) -> Option<Vec<(Vec<u8>, Vec<u8>)>> {
        let block_bytes = block_tokens.checked_mul(4)?;
        if block_bytes == 0 {
            return None;
        }
        raw.chunks(block_bytes)
            .map(|block| self.encode_indexed(block))
            .collect()
    }

    pub(crate) fn encode_frame(
        &self,
        raw: &[u8],
        block_tokens: usize,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        let block_bytes = block_tokens.checked_mul(4)?;
        if block_bytes == 0 {
            return None;
        }
        let mut data = Vec::with_capacity(raw.len());
        let mut entries = Vec::new();
        let mut tokens = 0u64;
        for block in raw.chunks(block_bytes) {
            let (payload, positions) = self.encode_indexed(block)?;
            data.extend_from_slice(&payload);
            let payload_end = data.len() as u64;
            data.extend_from_slice(&positions);
            tokens += (block.len() / 4) as u64;
            for value in [tokens, payload_end, data.len() as u64] {
                entries.extend_from_slice(&value.to_le_bytes());
            }
        }
        Some((data, entries))
    }

    fn encode_inner<const INDEX: bool>(&self, raw: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        if raw.len() % 4 != 0 {
            return None;
        }
        let capacity = (raw.len() / 4).checked_mul(19)?.checked_add(7)? / 8;
        let mut output = Vec::with_capacity(capacity);
        let mut checkpoints = if INDEX {
            Vec::with_capacity((raw.len() / 4).div_ceil(256).checked_mul(4)?)
        } else {
            Vec::new()
        };
        let mut reservoir = 0u64;
        let mut bits = 0usize;
        let lookup = |word: &[u8]| {
            let id = u32::from_le_bytes(word.try_into().unwrap());
            self.codes
                .get(id as usize)
                .map(|&(code, width)| (u64::from(code), usize::from(width)))
        };
        // Checkpoints cannot cut a group: begin a new group at each 256-token span.
        for checkpoint in raw.chunks(256 * 4) {
            if INDEX {
                checkpoints.extend_from_slice(
                    &u32::try_from(output.len().checked_mul(8)?.checked_add(bits)?)
                        .ok()?
                        .to_le_bytes(),
                );
            }
            let mut triples = checkpoint.chunks_exact(12);
            for triple in &mut triples {
                let (a, a_width) = lookup(&triple[..4])?;
                let (b, b_width) = lookup(&triple[4..8])?;
                let (c, c_width) = lookup(&triple[8..])?;
                let code = (a << (b_width + c_width)) | (b << c_width) | c;
                Self::append_code(
                    &mut output,
                    &mut reservoir,
                    &mut bits,
                    code,
                    a_width + b_width + c_width,
                );
            }
            for word in triples.remainder().chunks_exact(4) {
                let (code, width) = lookup(word)?;
                Self::append_code(&mut output, &mut reservoir, &mut bits, code, width);
            }
        }
        if bits != 0 {
            let padded = reservoir << (64 - bits);
            output.extend_from_slice(&padded.to_be_bytes()[..bits.div_ceil(8)]);
        }
        Some((output, checkpoints))
    }

    #[inline]
    fn append_code(
        output: &mut Vec<u8>,
        reservoir: &mut u64,
        bits: &mut usize,
        code: u64,
        width: usize,
    ) {
        let free = 64 - *bits;
        if width >= free {
            // A triple is at most 57 bits, so neither shift can reach 64.
            let remaining = width - free;
            *reservoir = (*reservoir << free) | (code >> remaining);
            output.extend_from_slice(&reservoir.to_be_bytes());
            *reservoir = code & ((1u64 << remaining) - 1);
            *bits = remaining;
        } else {
            *reservoir = (*reservoir << width) | code;
            *bits += width;
        }
    }

    pub(crate) fn decode(&self, data: &[u8], count: usize) -> Option<Vec<u32>> {
        self.decode_at(data, count, 0, 0, true, Vec::new())
    }

    pub(crate) fn decode_range(
        &self,
        data: &[u8],
        checkpoints: &[u8],
        tokens: usize,
        start: usize,
        count: usize,
    ) -> Option<Vec<u32>> {
        self.append_range(
            &IndexedRange {
                data,
                checkpoints,
                tokens,
                start,
                count,
            },
            Vec::new(),
        )
    }

    pub(crate) fn decode_ranges(&self, ranges: &[IndexedRange<'_>]) -> Option<Vec<u32>> {
        ranges
            .iter()
            .try_fold(Vec::new(), |output, range| self.append_range(range, output))
    }

    fn append_range(&self, range: &IndexedRange<'_>, output: Vec<u32>) -> Option<Vec<u32>> {
        let IndexedRange {
            data,
            checkpoints,
            tokens,
            start,
            count,
        } = *range;
        let end = start.checked_add(count)?;
        if end > tokens {
            return None;
        }
        if count == 0 {
            return Some(output);
        }
        let begin = (start / 256).checked_mul(4)?;
        let position = u32::from_le_bytes(
            checkpoints
                .get(begin..begin.checked_add(4)?)?
                .try_into()
                .ok()?,
        ) as usize;
        self.decode_at(data, count, position, start % 256, end == tokens, output)
    }

    fn decode_at(
        &self,
        data: &[u8],
        count: usize,
        mut position: usize,
        skip: usize,
        exact_end: bool,
        mut output: Vec<u32>,
    ) -> Option<Vec<u32>> {
        let total_bits = data.len().checked_mul(8)?;
        let steps = count.checked_add(skip)?;
        if steps > total_bits.checked_sub(position)? / self.widths[0] {
            return None;
        }
        output.reserve(count);
        for step in 0..steps {
            let begin = position / 8;
            let word = if data.len() - begin >= 4 {
                u32::from_be_bytes(data[begin..begin + 4].try_into().unwrap())
            } else {
                let mut bytes = [0; 4];
                bytes[..data.len() - begin].copy_from_slice(&data[begin..]);
                u32::from_be_bytes(bytes)
            } << (position % 8);
            // The leading code bits select one of three prefix branches.
            // Keep each branch's width and dictionary base independent of the
            // decoded payload instead of indexing four tiny tables per token.
            let (width, value) = if word & 0x8000_0000 == 0 {
                let width = self.widths[0];
                let index = (word >> (32 - width)) & self.masks[0];
                (width, self.order[index as usize])
            } else if word & 0x4000_0000 == 0 {
                let width = self.widths[2];
                let index = (word >> (32 - width)) & self.masks[2];
                (width, self.order[self.bases[2] + index as usize])
            } else {
                (19, (word >> 13) & 0x1ffff)
            };
            if step >= skip {
                output.push(value);
            }
            position += width;
            if position > total_bits {
                return None;
            }
        }
        if exact_end {
            if position.div_ceil(8) != data.len()
                || (position % 8 != 0 && data.last()? & ((1 << (8 - position % 8)) - 1) != 0)
            {
                return None;
            }
        }
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(ids: &[u32]) -> Vec<u8> {
        ids.iter().flat_map(|id| id.to_le_bytes()).collect()
    }

    fn oracle(ids: &[u32], order: &[u32], a: usize, b: usize) -> Vec<u8> {
        let mut bits = Vec::new();
        for &id in ids {
            let (value, width) = match order.iter().position(|&v| v == id) {
                Some(i) if i < 1 << a => (i as u32, a + 1),
                Some(i) => ((2 << b) | (i - (1 << a)) as u32, b + 2),
                None => ((3 << 17) | id, 19),
            };
            bits.extend((0..width).rev().map(|shift| ((value >> shift) & 1) as u8));
        }
        let mut bytes = vec![0; bits.len().div_ceil(8)];
        for (i, bit) in bits.into_iter().enumerate() {
            bytes[i / 8] |= bit << (7 - i % 8);
        }
        bytes
    }

    #[test]
    fn branch_ranges_match_original_values_at_every_checkpoint_offset() {
        for (a, b) in [(1, 2), (5, 10), (8, 14)] {
            let order: Vec<_> = (0..(1 << a) + (1 << b)).collect();
            let codec = Codec::new(&order, a, b).unwrap();
            let ids: Vec<u32> = (0..1793).map(|i| (i * 503) % 131072).collect();
            let (data, checkpoints) = codec.encode_indexed(&raw(&ids)).unwrap();
            for start in 0..512 {
                let count = 777;
                assert_eq!(
                    codec.decode_range(&data, &checkpoints, ids.len(), start, count),
                    Some(ids[start..start + count].to_vec())
                );
            }
            let ranges = [(257, 511), (0, 1), (1793, 0), (1792, 1), (511, 777)];
            let commands: Vec<_> = ranges
                .iter()
                .map(|&(start, count)| IndexedRange {
                    data: &data,
                    checkpoints: &checkpoints,
                    tokens: ids.len(),
                    start,
                    count,
                })
                .collect();
            let expected: Vec<_> = ranges
                .iter()
                .flat_map(|&(start, count)| ids[start..start + count].iter().copied())
                .collect();
            assert_eq!(codec.decode_ranges(&commands), Some(expected));
        }
    }

    #[test]
    fn branch_ranges_reject_truncation_in_prefix_and_output_and_bad_padding() {
        let codec = Codec::new(&(0..1056).collect::<Vec<_>>(), 5, 10).unwrap();
        let ids: Vec<u32> = (0..769).map(|i| (i * 503) % 131072).collect();
        let (data, checkpoints) = codec.encode_indexed(&raw(&ids)).unwrap();
        for end in [0, 1, data.len() / 2] {
            assert!(
                codec
                    .decode_range(&data[..end], &checkpoints, ids.len(), 127, 600)
                    .is_none()
            );
        }
        let mut tail = data.clone();
        tail.push(0);
        assert!(
            codec
                .decode_range(&tail, &checkpoints, ids.len(), 255, 514)
                .is_none()
        );
        let good = IndexedRange {
            data: &data,
            checkpoints: &checkpoints,
            tokens: ids.len(),
            start: 257,
            count: 300,
        };
        let bad = IndexedRange {
            data: &tail,
            checkpoints: &checkpoints,
            tokens: ids.len(),
            start: 255,
            count: 514,
        };
        assert!(codec.decode_ranges(&[good, bad]).is_none());
    }

    #[test]
    fn all_tiers_and_word_tails_match_independent_bits() {
        let order: Vec<_> = (0..1056).map(|i| (i * 37) % 131072).collect();
        let codec = Codec::new(&order, 5, 10).unwrap();
        for count in [0, 1, 2, 7, 8, 63, 64, 65, 8191, 8192, 8193] {
            let ids: Vec<_> = (0..count)
                .map(|i| [order[i % 32], order[32 + i % 1024], 131071][i % 3])
                .collect();
            let encoded = codec.encode(&raw(&ids)).unwrap();
            assert_eq!(encoded, oracle(&ids, &order, 5, 10));
            assert_eq!(codec.decode(&encoded, ids.len()).unwrap(), ids);
        }
    }

    #[test]
    fn every_original_id_round_trips() {
        let order: Vec<_> = (0..1056).rev().collect();
        let codec = Codec::new(&order, 5, 10).unwrap();
        let ids: Vec<_> = (0..131072).collect();
        let bytes = codec.encode(&raw(&ids)).unwrap();
        assert_eq!(codec.decode(&bytes, ids.len()).unwrap(), ids);
    }

    #[test]
    fn frame_blocks_match_separate_indexed_calls() {
        let codec = Codec::new(&(0..1056).rev().collect::<Vec<_>>(), 5, 10).unwrap();
        let ids: Vec<_> = (0..131072).collect();
        let bytes = raw(&ids);
        for block_tokens in [1, 255, 256, 8192, 131071, 131072, 131073] {
            let expected: Vec<_> = bytes
                .chunks(block_tokens * 4)
                .map(|block| codec.encode_indexed(block).unwrap())
                .collect();
            assert_eq!(codec.encode_blocks(&bytes, block_tokens).unwrap(), expected);
        }
        assert_eq!(codec.encode_blocks(&[], 8192), Some(vec![]));
        assert!(codec.encode_blocks(&bytes, 0).is_none());
        assert!(codec.encode_blocks(&[], usize::MAX).is_none());
        assert!(codec.encode_blocks(&[0], 8192).is_none());
        assert!(codec.encode_blocks(&raw(&[0, 131072]), 1).is_none());
    }

    #[test]
    fn word_spills_and_final_padding_match_the_bit_oracle() {
        let order: Vec<_> = (0..1056).collect();
        let codec = Codec::new(&order, 5, 10).unwrap();
        let pattern = [0, 31, 32, 1055, 65536, 131071];
        for rotation in 0..pattern.len() {
            for count in 0..257 {
                let ids: Vec<_> = (0..count)
                    .map(|index| pattern[(index + rotation) % pattern.len()])
                    .collect();
                let expected = oracle(&ids, &order, 5, 10);
                assert_eq!(codec.encode(&raw(&ids)).unwrap(), expected);
                let (bytes, positions) = codec.encode_indexed(&raw(&ids)).unwrap();
                assert_eq!(bytes, expected);
                assert_eq!(
                    codec
                        .decode_range(&bytes, &positions, count, 0, count)
                        .unwrap(),
                    ids
                );
            }
        }
    }

    #[test]
    fn triple_tails_and_checkpoint_cuts_match_each_legal_width_extreme() {
        for (a, b) in [(1, 2), (5, 10), (8, 14)] {
            let small = 1usize << a;
            let order: Vec<_> = (0..small + (1usize << b)).map(|i| i as u32).collect();
            let codec = Codec::new(&order, a, b).unwrap();
            let pattern = [
                0,
                (small - 1) as u32,
                small as u32,
                *order.last().unwrap(),
                131071,
            ];
            for count in [0, 1, 2, 3, 4, 254, 255, 256, 257, 258, 513] {
                let ids: Vec<_> = (0..count).map(|i| pattern[i % pattern.len()]).collect();
                let (encoded, positions) = codec.encode_indexed(&raw(&ids)).unwrap();
                assert_eq!(encoded, oracle(&ids, &order, a as usize, b as usize));
                assert_eq!(codec.decode(&encoded, count).unwrap(), ids);
                for start in [0, 1, 2, 254, 255, 256, 257, 512] {
                    if start < count {
                        assert_eq!(
                            codec
                                .decode_range(&encoded, &positions, count, start, count - start)
                                .unwrap(),
                            ids[start..],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_ids_in_each_triple_position_return_no_output() {
        let codec = Codec::new(&(0..1056).collect::<Vec<_>>(), 5, 10).unwrap();
        for index in [0, 1, 2, 3, 254, 255, 256, 257] {
            let mut ids = vec![31; 258];
            ids[index] = 131072;
            assert!(codec.encode(&raw(&ids)).is_none());
            assert!(codec.encode_indexed(&raw(&ids)).is_none());
            assert!(codec.encode_blocks(&raw(&ids), 256).is_none());
        }
    }

    #[test]
    fn invalid_tables_are_rejected() {
        for (order, a, b) in [
            (vec![], 0, 1),
            (vec![], 5, 10),
            (vec![0; 1056], 5, 10),
            ((0..1056).collect(), 10, 5),
        ] {
            assert!(Codec::new(&order, a, b).is_none());
        }
        let mut order: Vec<_> = (0..1056).collect();
        order[1] = 131072;
        assert!(Codec::new(&order, 5, 10).is_none());
    }

    #[test]
    fn block_errors_and_empty_blocks_are_preserved() {
        let codec = Codec::new(&(0..1056).collect::<Vec<_>>(), 5, 10).unwrap();
        let bytes = codec.encode(&raw(&[0, 32, 131071])).unwrap();
        assert!(codec.decode(&bytes[..bytes.len() - 1], 3).is_none());
        assert!(
            codec
                .decode(&[bytes.as_slice(), &[0]].concat(), 3)
                .is_none()
        );
        assert!(codec.decode(&bytes, usize::MAX).is_none());
        assert!(codec.encode(&[0]).is_none());
        assert!(codec.encode(&raw(&[131072])).is_none());
        assert_eq!(codec.decode(&[], 0), Some(vec![]));
    }

    #[test]
    fn indexed_ranges_match_original_bits_and_positions() {
        let order: Vec<_> = (0..1056).map(|i| (i * 37) % 131072).collect();
        let codec = Codec::new(&order, 5, 10).unwrap();
        let ids: Vec<_> = (0..8193)
            .map(|i| [order[i % 32], order[32 + i % 1024], 131071][i % 3])
            .collect();
        let (bytes, positions) = codec.encode_indexed(&raw(&ids)).unwrap();
        assert_eq!(bytes, oracle(&ids, &order, 5, 10));
        for (index, bytes) in positions.chunks_exact(4).enumerate() {
            let position = u32::from_le_bytes(bytes.try_into().unwrap());
            let prefix = &ids[..index * 256];
            let expected: usize = prefix
                .iter()
                .map(|id| match order.iter().position(|v| v == id) {
                    Some(i) if i < 32 => 6,
                    Some(_) => 12,
                    None => 19,
                })
                .sum();
            assert_eq!(position as usize, expected);
        }
        for start in [0, 1, 255, 256, 257, 2047, 4095, 8192, 8193] {
            for count in [0, 1, 3, 2048] {
                if start + count <= ids.len() {
                    assert_eq!(
                        codec
                            .decode_range(&bytes, &positions, ids.len(), start, count)
                            .unwrap(),
                        ids[start..start + count]
                    );
                }
            }
        }
    }

    #[test]
    fn indexed_ranges_reject_invalid_extents() {
        let codec = Codec::new(&(0..1056).collect::<Vec<_>>(), 5, 10).unwrap();
        let ids: Vec<_> = (0..1025).map(|i| (i * 503) % 131072).collect();
        let (bytes, positions) = codec.encode_indexed(&raw(&ids)).unwrap();
        assert!(codec.decode_range(&bytes, &[], ids.len(), 1, 1).is_none());
        assert!(
            codec
                .decode_range(&bytes, &positions, ids.len(), 1025, 1)
                .is_none()
        );
        assert!(
            codec
                .decode_range(&bytes, &positions, ids.len(), usize::MAX, 2)
                .is_none()
        );
        assert!(
            codec
                .decode_range(&bytes, &[255; 4], ids.len(), 0, 1)
                .is_none()
        );
        assert!(
            codec
                .decode_range(&bytes[..1], &positions, ids.len(), 0, 10)
                .is_none()
        );
        assert_eq!(codec.encode_indexed(&[]), Some((vec![], vec![])));
        assert_eq!(codec.decode_range(&[], &[], 0, 0, 0), Some(vec![]));
    }

    #[test]
    fn grouped_ranges_preserve_order_and_discard_partial_results_on_error() {
        let codec = Codec::new(&(0..1056).collect::<Vec<_>>(), 5, 10).unwrap();
        let ids: Vec<_> = (0..8193).map(|i| (i * 503) % 131072).collect();
        let (data, checkpoints) = codec.encode_indexed(&raw(&ids)).unwrap();
        let mut ranges: Vec<_> = [(255, 1024), (0, 0), (8192, 1), (0, 256)]
            .into_iter()
            .map(|(start, count)| IndexedRange {
                data: &data,
                checkpoints: &checkpoints,
                tokens: ids.len(),
                start,
                count,
            })
            .collect();
        let expected: Vec<_> = ranges
            .iter()
            .flat_map(|r| ids[r.start..r.start + r.count].iter().copied())
            .collect();
        assert_eq!(codec.decode_ranges(&ranges), Some(expected));
        assert_eq!(codec.decode_ranges(&[]), Some(vec![]));
        ranges.push(IndexedRange {
            data: &data,
            checkpoints: &checkpoints,
            tokens: ids.len(),
            start: usize::MAX,
            count: 2,
        });
        assert!(codec.decode_ranges(&ranges).is_none());
    }

    #[test]
    fn contiguous_frames_preserve_blocks_and_reject_partial_input() {
        let codec = Codec::new(&(0..1056).collect::<Vec<_>>(), 5, 10).unwrap();
        let ids: Vec<_> = (0..16385).map(|i| (i * 503) % 131072).collect();
        let input = raw(&ids);
        for size in [1, 255, 256, 257, 8192, 16385, 16386] {
            let expected = codec.encode_blocks(&input, size).unwrap();
            let (data, entries) = codec.encode_frame(&input, size).unwrap();
            assert_eq!(entries.len(), expected.len() * 24);
            let mut begin = 0;
            let mut tokens = 0;
            for (entry, (payload, positions)) in entries.chunks_exact(24).zip(expected) {
                let count = u64::from_le_bytes(entry[..8].try_into().unwrap()) as usize;
                let payload_end = u64::from_le_bytes(entry[8..16].try_into().unwrap()) as usize;
                let end = u64::from_le_bytes(entry[16..].try_into().unwrap()) as usize;
                assert_eq!(count - tokens, size.min(ids.len() - tokens));
                assert_eq!(&data[begin..payload_end], &payload);
                assert_eq!(&data[payload_end..end], &positions);
                begin = end;
                tokens = count;
            }
            assert_eq!(begin, data.len());
            assert_eq!(tokens, ids.len());
        }
        assert_eq!(codec.encode_frame(&[], 1), Some((vec![], vec![])));
        assert!(codec.encode_frame(&input, 0).is_none());
        assert!(codec.encode_frame(&input, usize::MAX).is_none());
        assert!(
            codec
                .encode_frame(&input[..input.len() - 1], 8192)
                .is_none()
        );
        assert!(codec.encode_frame(&raw(&[1, 2, 131072]), 1).is_none());
    }
}
