use std::ops::Index;

use rustc_hash::FxHashMap as HashMap;

use crate::Rank;

pub(crate) trait RankLookup: for<'a> Index<&'a [u8], Output = Rank> {
    fn get(&self, bytes: &[u8]) -> Option<&Rank>;

    #[inline]
    fn get_span(&self, source: &[u8], start: usize, end: usize) -> Option<&Rank> {
        self.get(&source[start..end])
    }
}

impl RankLookup for HashMap<Vec<u8>, Rank> {
    #[inline]
    fn get(&self, bytes: &[u8]) -> Option<&Rank> {
        HashMap::get(self, bytes)
    }
}

// The high byte records the length, preserving binary keys with trailing NULs.
#[inline(always)]
fn short_key(bytes: &[u8]) -> u64 {
    let word = match bytes {
        [] => 0,
        [a] => u64::from(*a),
        [a, b] => u64::from(u16::from_le_bytes([*a, *b])),
        [a, b, c] => u64::from(u16::from_le_bytes([*a, *b])) | (u64::from(*c) << 16),
        [a, b, c, d] => u64::from(u32::from_le_bytes([*a, *b, *c, *d])),
        [a, b, c, d, e] => u64::from(u32::from_le_bytes([*a, *b, *c, *d])) | (u64::from(*e) << 32),
        [a, b, c, d, e, f] => {
            u64::from(u32::from_le_bytes([*a, *b, *c, *d]))
                | (u64::from(u16::from_le_bytes([*e, *f])) << 32)
        }
        [a, b, c, d, e, f, g] => {
            u64::from(u32::from_le_bytes([*a, *b, *c, *d]))
                | (u64::from(u16::from_le_bytes([*e, *f])) << 32)
                | (u64::from(*g) << 48)
        }
        _ => unreachable!(),
    };
    word | ((bytes.len() as u64) << 56)
}

// Read only within the immutable parent slice. Bytes outside the queried span
// are masked away, so adjacent text cannot affect the existing packed key.
#[inline(always)]
fn short_key_window(source: &[u8], start: usize, length: usize) -> u64 {
    if length == 0 {
        return 0;
    }
    let window = start.min(source.len() - 8);
    let word = u64::from_le_bytes(source[window..window + 8].try_into().unwrap());
    let shift = (start - window) * 8;
    let mask = (1_u64 << (length * 8)) - 1;
    ((word >> shift) & mask) | ((length as u64) << 56)
}

// Lengths through seven use their existing high-byte tag. A complete
// eight-byte word is admitted only when its high byte is outside those tags.
const SHORT_WINDOW_MASKS: [u64; 9] = [
    0,
    0xff,
    0xffff,
    0xffffff,
    0xffffffff,
    0xffffffffff,
    0xffffffffffff,
    0xffffffffffffff,
    u64::MAX,
];

#[inline(always)]
fn short_or_eight_key_window(source: &[u8], start: usize, length: usize) -> u64 {
    let window = start.min(source.len() - 8);
    let word = u64::from_le_bytes(source[window..window + 8].try_into().unwrap());
    // Empty spans may start at EOF. Their zero mask removes the loaded word
    // even when the defined wrapping shift is 64.
    let word = word.wrapping_shr(((start - window) * 8) as u32);
    (word & SHORT_WINDOW_MASKS[length]) | (((length & 7) as u64) << 56)
}

#[repr(C)]
#[derive(Clone, Copy, Eq, Hash)]
struct MediumKey {
    first: u64,
    tail: u64,
}

impl PartialEq for MediumKey {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};

            // SSE2 is baseline on x86_64. repr(C) with two u64 fields provides
            // exactly 16 initialized bytes; loadu does not require 16-byte alignment.
            unsafe {
                let left = _mm_loadu_si128(std::ptr::from_ref(self).cast());
                let right = _mm_loadu_si128(std::ptr::from_ref(other).cast());
                _mm_movemask_epi8(_mm_cmpeq_epi8(left, right)) == 0xffff
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.first == other.first && self.tail == other.tail
        }
    }
}

impl MediumKey {
    #[inline(always)]
    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            first: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            tail: short_key(&bytes[8..]),
        }
    }
}

#[derive(Clone)]
pub(crate) struct EncoderRanks {
    short: HashMap<u64, Rank>,
    medium: HashMap<MediumKey, Rank>,
    long: HashMap<Vec<u8>, Rank>,
}

impl From<HashMap<Vec<u8>, Rank>> for EncoderRanks {
    fn from(ranks: HashMap<Vec<u8>, Rank>) -> Self {
        let mut counts = [0; 3];
        for bytes in ranks.keys() {
            counts[match bytes.len() {
                0..=7 => 0,
                8 if bytes[7] > 7 => 0,
                8..=15 => 1,
                _ => 2,
            }] += 1;
        }
        let mut short = HashMap::with_capacity_and_hasher(counts[0], Default::default());
        let mut medium = HashMap::with_capacity_and_hasher(counts[1], Default::default());
        let mut long = HashMap::with_capacity_and_hasher(counts[2], Default::default());
        for (bytes, rank) in ranks {
            match bytes.len() {
                0..=7 => short.insert(short_key(&bytes), rank),
                8 if bytes[7] > 7 => {
                    short.insert(u64::from_le_bytes(bytes[..8].try_into().unwrap()), rank)
                }
                8..=15 => medium.insert(MediumKey::from_bytes(&bytes), rank),
                _ => long.insert(bytes, rank),
            };
        }
        Self {
            short,
            medium,
            long,
        }
    }
}

impl EncoderRanks {
    #[inline(always)]
    pub(crate) fn get(&self, bytes: &[u8]) -> Option<&Rank> {
        match bytes.len() {
            0..=7 => self.short.get(&short_key(bytes)),
            8 if bytes[7] > 7 => self
                .short
                .get(&u64::from_le_bytes(bytes[..8].try_into().unwrap())),
            8..=15 => self.medium.get(&MediumKey::from_bytes(bytes)),
            _ => self.long.get(bytes),
        }
    }

    #[inline(always)]
    pub(crate) fn get_span(&self, source: &[u8], start: usize, end: usize) -> Option<&Rank> {
        let bytes = &source[start..end];
        if source.len() < 8 {
            return self.get(bytes);
        }
        match bytes.len() {
            0..=8 => {
                let key = short_or_eight_key_window(source, start, bytes.len());
                // For lengths 0..7 the added term is at least 64; for length
                // eight it is zero. Only an eight-byte key ending in 0..7
                // reaches the medium map. Combining the values avoids another
                // unpredictable length branch in this shared query path.
                if (key >> 56) + (((bytes.len() ^ 8) * 8) as u64) < 8 {
                    self.medium.get(&MediumKey {
                        first: key,
                        tail: 0,
                    })
                } else {
                    self.short.get(&key)
                }
            }
            9..=15 => self.medium.get(&MediumKey {
                first: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                tail: short_key_window(source, start + 8, bytes.len() - 8),
            }),
            _ => self.long.get(bytes),
        }
    }

    pub(crate) fn contains_key(&self, bytes: &[u8]) -> bool {
        self.get(bytes).is_some()
    }
}

impl RankLookup for EncoderRanks {
    #[inline(always)]
    fn get(&self, bytes: &[u8]) -> Option<&Rank> {
        self.get(bytes)
    }

    #[inline(always)]
    fn get_span(&self, source: &[u8], start: usize, end: usize) -> Option<&Rank> {
        self.get_span(source, start, end)
    }
}

impl Index<&[u8]> for EncoderRanks {
    type Output = Rank;

    #[inline(always)]
    fn index(&self, bytes: &[u8]) -> &Rank {
        self.get(bytes).expect("no entry found for key")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_window_matches_all_short_lengths_and_end_alignments() {
        for changed in 0..32 {
            for byte in 0..=255_u8 {
                let mut source = [0_u8; 32];
                source[changed] = byte;
                for start in 0..source.len() {
                    for length in 1..=7.min(source.len() - start) {
                        assert_eq!(
                            short_key_window(&source, start, length),
                            short_key(&source[start..start + length])
                        );
                    }
                }
            }
        }
        assert_eq!(short_key_window(&[0; 8], 8, 0), short_key(&[]));
    }

    #[test]
    fn span_lookup_matches_original_for_binary_sparse_and_full_ranges() {
        let mut original = HashMap::default();
        for seed in 0..8 {
            let source: Vec<u8> = (0..40).map(|i| ((i * 31 + seed * 7) % 256) as u8).collect();
            for start in 0..source.len() {
                for end in start..=source.len() {
                    if (start + end + seed) % 3 != 0 {
                        original.insert(
                            source[start..end].to_vec(),
                            (start * 97 + end + seed) as Rank,
                        );
                    }
                }
            }
        }
        original.insert(Vec::new(), Rank::MAX);
        let ranks = EncoderRanks::from(original.clone());
        for seed in 0..16 {
            let source: Vec<u8> = (0..40).map(|i| ((i * 31 + seed * 7) % 256) as u8).collect();
            for parent_length in 0..=source.len() {
                let parent = &source[..parent_length];
                for start in 0..=parent_length {
                    for end in start..=parent_length {
                        assert_eq!(
                            ranks.get_span(parent, start, end),
                            original.get(&parent[start..end])
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn parent_bytes_outside_key_do_not_change_lookup() {
        let mut original = HashMap::default();
        original.insert(b"abc".to_vec(), Rank::MAX);
        original.insert(b"abcdefghijklmno".to_vec(), 17);
        let ranks = EncoderRanks::from(original);
        for prefix in 0..20 {
            for suffix in 0..20 {
                for key in [b"abc".as_slice(), b"abcdefghijklmno".as_slice()] {
                    let mut source = vec![255; prefix];
                    source.extend(key);
                    source.extend(vec![0; suffix]);
                    assert_eq!(
                        ranks.get_span(&source, prefix, prefix + key.len()),
                        ranks.get(key)
                    );
                }
            }
        }
    }

    #[test]
    fn eighth_byte_keys_are_disjoint_from_length_tags() {
        let mut original = HashMap::default();
        for len in 0..=7 {
            for byte in 0..=255_u8 {
                original.insert(vec![byte; len], original.len() as Rank);
            }
        }
        for terminal in 0..=255_u8 {
            for first in [0, 1, 7, 8, 127, 128, 255] {
                let mut key = vec![0; 8];
                key[0] = first;
                key[7] = terminal;
                original.insert(key, original.len() as Rank);
            }
        }
        original.insert(vec![255; 8], Rank::MAX);
        let expected_short = original
            .keys()
            .filter(|k| k.len() <= 7 || (k.len() == 8 && k[7] > 7))
            .count();
        let expected_medium = original.len() - expected_short;
        let encoded = EncoderRanks::from(original.clone());
        assert_eq!(encoded.short.len(), expected_short);
        assert_eq!(encoded.medium.len(), expected_medium);
        for (key, rank) in original {
            assert_eq!(encoded.get(&key), Some(&rank));
            for prefix in 0..=8 {
                for suffix in 0..=8 {
                    let mut parent = vec![255; prefix];
                    parent.extend_from_slice(&key);
                    parent.extend(vec![0; suffix]);
                    assert_eq!(
                        encoded.get_span(&parent, prefix, prefix + key.len()),
                        Some(&rank)
                    );
                }
            }
        }
        for terminal in 0..=255_u8 {
            let mut key = vec![43; 8];
            key[7] = terminal;
            assert_eq!(encoded.get(&key), None);
        }
    }

    #[test]
    fn medium_equality_checks_both_words() {
        let key = MediumKey { first: 7, tail: 11 };
        assert!(key == MediumKey { first: 7, tail: 11 });
        for bit in 0..64 {
            assert!(
                key != MediumKey {
                    first: 7 ^ (1 << bit),
                    tail: 11
                }
            );
            assert!(
                key != MediumKey {
                    first: 7,
                    tail: 11 ^ (1 << bit)
                }
            );
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    fn word_pair_reduces_slot_alignment() {
        assert_eq!(std::mem::size_of::<(MediumKey, Rank)>(), 24);
        assert_eq!(std::mem::align_of::<MediumKey>(), 8);
        assert_eq!(std::mem::size_of::<(u128, Rank)>(), 32);
    }

    #[test]
    fn binary_keys_lengths_and_every_medium_byte_are_exact() {
        let mut original = HashMap::default();
        original.insert(Vec::new(), Rank::MAX);
        for a in 0..=255_u8 {
            original.insert(vec![a], u32::from(a));
            for b in 0..=255_u8 {
                original.insert(vec![a, b], 256 + (u32::from(a) << 8) + u32::from(b));
            }
        }
        for len in 3..=128 {
            original.insert(vec![0; len], 100_000 + len as Rank);
        }
        for len in 8..=15 {
            for position in 0..len {
                for byte in 1..=255_u8 {
                    let mut bytes = vec![0; len];
                    bytes[position] = byte;
                    original.insert(
                        bytes,
                        200_000 + len as Rank * 4096 + position as Rank * 256 + Rank::from(byte),
                    );
                }
            }
        }
        let ranks = EncoderRanks::from(original.clone());
        assert_eq!(
            ranks.short.len() + ranks.medium.len() + ranks.long.len(),
            original.len()
        );
        for (bytes, rank) in &original {
            assert_eq!(ranks.get(bytes), Some(rank));
            let mut neighbor = bytes.clone();
            neighbor.push(0);
            assert_eq!(ranks.get(&neighbor), original.get(&neighbor));
        }
        let clone = ranks.clone();
        drop(ranks);
        for (bytes, rank) in original {
            assert_eq!(clone.get(&bytes), Some(&rank));
        }
    }

    #[test]
    fn both_merge_paths_match_byte_key_map() {
        for seed in 0..8_u32 {
            let mut original: HashMap<Vec<u8>, Rank> = (0..=255_u8)
                .map(|byte| (vec![byte], u32::from(byte)))
                .collect();
            for len in 2..=128 {
                let bytes = (0..len).map(|i| (i % 8) as u8).collect();
                original.insert(
                    bytes,
                    if len % 13 == 0 {
                        Rank::MAX
                    } else {
                        500 + (len as Rank * (seed + 1)) % 31
                    },
                );
            }
            let ranks = EncoderRanks::from(original.clone());
            for len in 1..=160 {
                let bytes: Vec<u8> = (0..len).map(|i| (i % 8) as u8).collect();
                assert_eq!(
                    crate::_byte_pair_encode(&bytes, &ranks),
                    crate::byte_pair_encode(&bytes, &original)
                );
            }
        }
    }
}
