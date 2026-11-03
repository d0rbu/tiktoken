use std::hash::Hasher;
use std::ops::Index;

use rustc_hash::{FxHashMap as HashMap, FxHasher};

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

// All four exact keys and their ranks share one cache line. Overflow keeps
// arbitrary vocabularies exact even if many keys select the same bucket.
#[repr(C, align(64))]
#[derive(Clone, Default)]
struct ShortBucket {
    keys: [u64; 4],
    ranks: [Rank; 4],
    occupied: u8,
}

impl ShortBucket {
    #[inline(always)]
    fn matches(&self, key: u64) -> u8 {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{
                _mm_and_si128, _mm_castsi128_pd, _mm_cmpeq_epi32, _mm_loadu_si128, _mm_movemask_pd,
                _mm_set1_epi64x, _mm_shuffle_epi32,
            };
            // SSE2 is baseline on x86-64. Both loads are wholly within the
            // four initialized u64 keys. Pair adjacent 32-bit comparisons so
            // each output bit tests the complete 64-bit key, including NULs.
            unsafe {
                let query = _mm_set1_epi64x(key as i64);
                let first = _mm_cmpeq_epi32(_mm_loadu_si128(self.keys.as_ptr().cast()), query);
                let last =
                    _mm_cmpeq_epi32(_mm_loadu_si128(self.keys.as_ptr().add(2).cast()), query);
                let first = _mm_and_si128(first, _mm_shuffle_epi32(first, 0xb1));
                let last = _mm_and_si128(last, _mm_shuffle_epi32(last, 0xb1));
                ((_mm_movemask_pd(_mm_castsi128_pd(first))
                    | (_mm_movemask_pd(_mm_castsi128_pd(last)) << 2)) as u8)
                    & self.occupied
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut matched = 0;
            for (index, &stored) in self.keys.iter().enumerate() {
                if stored == key {
                    matched |= 1 << index;
                }
            }
            matched & self.occupied
        }
    }
}

#[derive(Clone)]
struct ShortRanks {
    buckets: Box<[ShortBucket]>,
    overflow: HashMap<u64, Rank>,
}

impl ShortRanks {
    fn with_capacity(count: usize) -> Self {
        let buckets = count.div_ceil(2).max(1).next_power_of_two();
        Self {
            buckets: vec![ShortBucket::default(); buckets].into_boxed_slice(),
            overflow: HashMap::default(),
        }
    }

    #[inline(always)]
    fn bucket(&self, key: u64) -> usize {
        let mut hasher = FxHasher::default();
        hasher.write_u64(key);
        hasher.finish() as usize & (self.buckets.len() - 1)
    }

    fn insert(&mut self, key: u64, rank: Rank) {
        let index = self.bucket(key);
        let bucket = &mut self.buckets[index];
        let slot = bucket.occupied.count_ones() as usize;
        if slot < 4 {
            bucket.keys[slot] = key;
            bucket.ranks[slot] = rank;
            bucket.occupied |= 1 << slot;
        } else {
            self.overflow.insert(key, rank);
        }
    }

    #[inline(always)]
    fn get(&self, key: &u64) -> Option<&Rank> {
        let bucket = &self.buckets[self.bucket(*key)];
        let matched = bucket.matches(*key);
        if matched != 0 {
            Some(&bucket.ranks[matched.trailing_zeros() as usize])
        } else {
            self.overflow.get(key)
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.buckets
            .iter()
            .map(|b| b.occupied.count_ones() as usize)
            .sum::<usize>()
            + self.overflow.len()
    }
}

#[derive(Clone)]
pub(crate) struct EncoderRanks {
    short: ShortRanks,
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
        let mut short = ShortRanks::with_capacity(counts[0]);
        let mut medium = HashMap::with_capacity_and_hasher(counts[1], Default::default());
        let mut long = HashMap::with_capacity_and_hasher(counts[2], Default::default());
        for (bytes, rank) in ranks {
            match bytes.len() {
                0..=7 => short.insert(short_key(&bytes), rank),
                8..=15 => {
                    medium.insert(MediumKey::from_bytes(&bytes), rank);
                }
                _ => {
                    long.insert(bytes, rank);
                }
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

    #[inline(always)]
    pub(crate) fn get_span(&self, source: &[u8], start: usize, end: usize) -> Option<&Rank> {
        let bytes = &source[start..end];
        if source.len() < 8 {
            return self.get(bytes);
        }
        match bytes.len() {
            0..=7 => self
                .short
                .get(&short_key_window(source, start, bytes.len())),
            8..=15 => self.medium.get(&MediumKey {
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
    fn short_bucket_has_one_cache_line_and_exact_full_word_matches() {
        assert_eq!(std::mem::size_of::<ShortBucket>(), 64);
        assert_eq!(std::mem::align_of::<ShortBucket>(), 64);
        let mut bucket = ShortBucket {
            keys: [0, 0xffff_ffff, 0xffff_ffff_0000_0000, u64::MAX],
            ..ShortBucket::default()
        };
        for occupied in 0..16 {
            bucket.occupied = occupied;
            for key in bucket.keys {
                for bit in 0..=64 {
                    let query = if bit == 64 { key } else { key ^ (1 << bit) };
                    let expected = bucket.keys.iter().enumerate().fold(0, |mask, (i, &k)| {
                        mask | if k == query { (1 << i) & occupied } else { 0 }
                    });
                    assert_eq!(bucket.matches(query), expected);
                }
            }
        }
    }

    #[test]
    fn short_bucket_overflow_clone_and_empty_key_are_exact() {
        let mut ranks = ShortRanks::with_capacity(0);
        assert_eq!(ranks.get(&0), None);
        for key in 0..12_u64 {
            ranks.insert(key, if key == 3 { Rank::MAX } else { key as Rank });
        }
        assert_eq!(ranks.buckets.len(), 1);
        assert_eq!(ranks.overflow.len(), 8);
        assert_eq!(ranks.len(), 12);
        let cloned = ranks.clone();
        for key in 0..=12_u64 {
            let expected = if key == 12 {
                None
            } else if key == 3 {
                Some(Rank::MAX)
            } else {
                Some(key as Rank)
            };
            assert_eq!(ranks.get(&key).copied(), expected);
            assert_eq!(cloned.get(&key).copied(), expected);
        }
    }

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
