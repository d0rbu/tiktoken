//! Compact immutable vocabulary buckets, with complete byte equality.

use std::hash::BuildHasher;

use hashbrown::HashTable;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::Rank;

// Larger public vocabularies retain the original lookup. The RMCP family
// already bounds its complete vocabulary bytes below this size.
const MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Entry {
    start: u32,
    end: u32,
    rank: Rank,
}

#[derive(Clone)]
pub(crate) struct RankIndex {
    bytes: Vec<u8>,
    entries: HashTable<Entry>,
}

impl RankIndex {
    pub fn new(ranks: &FxHashMap<Vec<u8>, Rank>) -> Option<Self> {
        let size = ranks
            .keys()
            .try_fold(0_usize, |size, word| size.checked_add(word.len()))?;
        if size > MAX_BYTES {
            return None;
        }
        let mut bytes = Vec::with_capacity(size);
        let mut entries = HashTable::with_capacity(ranks.len());
        for (word, &rank) in ranks {
            let start = bytes.len() as u32;
            bytes.extend_from_slice(word);
            let entry = Entry {
                start,
                end: bytes.len() as u32,
                rank,
            };
            entries.insert_unique(FxBuildHasher.hash_one(word.as_slice()), entry, |entry| {
                FxBuildHasher.hash_one(&bytes[entry.start as usize..entry.end as usize])
            });
        }
        Some(Self { bytes, entries })
    }

    #[inline(always)]
    pub fn get(&self, word: &[u8]) -> Option<Rank> {
        self.entries
            .find(FxBuildHasher.hash_one(word), |entry| {
                &self.bytes[entry.start as usize..entry.end as usize] == word
            })
            .map(|entry| entry.rank)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::hash::BuildHasher;

    use rustc_hash::{FxBuildHasher, FxHashMap};

    use super::{MAX_BYTES, RankIndex};
    use crate::{CoreBPE, Rank};

    #[test]
    fn full_bytes_distinguish_colliding_hashes_and_missing_keys() {
        // These pinned Fx byte hashes collide: the first word cancels its
        // initial mixing operand. A hash match must never imply token equality.
        let prefix = 0x243f6a8885a308d3_u64.to_le_bytes();
        let first = [prefix.as_slice(), &[0; 8]].concat();
        let second = [prefix.as_slice(), &[1; 8]].concat();
        let absent = [prefix.as_slice(), &[2; 8]].concat();
        assert_eq!(
            FxBuildHasher.hash_one(first.as_slice()),
            FxBuildHasher.hash_one(second.as_slice())
        );
        let ranks = FxHashMap::from_iter([(first.clone(), 0), (second.clone(), Rank::MAX)]);
        let index = RankIndex::new(&ranks).unwrap();
        assert_eq!(index.get(&first), Some(0));
        assert_eq!(index.get(&second), Some(Rank::MAX));
        assert_eq!(index.get(&absent), None);
        let cloned = index.clone();
        drop(index);
        assert_eq!(cloned.get(&first), Some(0));
        assert_eq!(cloned.get(&second), Some(Rank::MAX));
    }

    #[test]
    fn byte_strings_and_prefixes_match_the_original_vocabulary() {
        let mut ranks = FxHashMap::from_iter([(Vec::new(), 1000)]);
        for byte in 0_u16..256 {
            ranks.insert(vec![byte as u8], byte as Rank);
            ranks.insert(vec![byte as u8, 0, 255], byte as Rank + 256);
        }
        let index = RankIndex::new(&ranks).unwrap();
        for (word, &rank) in &ranks {
            assert_eq!(index.get(word), Some(rank));
            let mut longer = word.clone();
            longer.push(1);
            assert_eq!(index.get(&longer), ranks.get(&longer).copied());
        }
        assert!(
            RankIndex::new(&FxHashMap::default())
                .unwrap()
                .get(b"")
                .is_none()
        );
    }

    #[test]
    fn larger_public_vocabulary_uses_the_original_map() {
        let mut ranks: FxHashMap<Vec<u8>, Rank> = (0_u16..256)
            .map(|byte| (vec![byte as u8], byte as Rank))
            .collect();
        ranks.insert(vec![b'a'; MAX_BYTES + 1], 1000);
        assert!(RankIndex::new(&ranks).is_none());
        let model = CoreBPE::new_internal(ranks, FxHashMap::default(), "a").unwrap();
        assert_eq!(model.encode("a", &HashSet::new()).unwrap().0, [97]);
        assert_eq!(model.count_tokens("a", &HashSet::new()).unwrap(), 1);
    }
}
