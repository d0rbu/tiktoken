use std::ops::Index;

use rustc_hash::FxHashMap as HashMap;

use crate::Rank;

pub(crate) trait RankLookup: for<'a> Index<&'a [u8], Output = Rank> {
    fn get(&self, bytes: &[u8]) -> Option<&Rank>;
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
            8..=15 => self.medium.get(&MediumKey::from_bytes(bytes)),
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
