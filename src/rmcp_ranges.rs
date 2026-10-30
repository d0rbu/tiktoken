//! One-call range index for the exact guarded RMCP pattern. No encoder state is
//! retained, and every other pattern keeps independent scalar counting.

use std::collections::HashSet;

use crate::{CoreBPE, EncodeError, _byte_pair_merge, byte_pair_encode};

impl CoreBPE {
    /// Visit the actual ordinary/special partitions, stopping when requested.
    /// A failed speculative visit is discarded in favor of the original API.
    fn visit_count_pieces(
        &self,
        text: &str,
        allowed: &HashSet<&str>,
        mut visit: impl FnMut(usize, &str, bool) -> bool,
    ) -> Result<(), EncodeError> {
        let regex = self._get_tl_special_regex();
        let mut start = 0;
        loop {
            let mut search = start;
            let next = loop {
                let found = regex
                    .find_from_pos(text, search)
                    .map_err(|error| EncodeError {
                        message: format!("Special regex error while indexing: {error}"),
                    })?;
                match found {
                    Some(m) if !allowed.contains(m.as_str()) => search = m.start() + 1,
                    _ => break found,
                }
            };
            let end = next.map_or(text.len(), |m| m.start());
            let mut cursor = start;
            for piece in self._ordinary_pieces(&text[start..end]) {
                let piece = piece.map_err(|error| EncodeError {
                    message: format!("Regex error while indexing: {error}"),
                })?;
                cursor += piece.len();
                if !visit(cursor, piece, false) {
                    return Ok(());
                }
            }
            if cursor != end {
                return Err(EncodeError {
                    message: "Incomplete guarded partition".to_owned(),
                });
            }
            match next {
                Some(m) => {
                    if !visit(m.end(), m.as_str(), true) {
                        return Ok(());
                    }
                    start = m.end();
                }
                None => return Ok(()),
            }
        }
    }

    fn count_index_piece(&self, piece: &[u8], special: bool) -> usize {
        if special || self.encoder.contains_key(piece) {
            1
        } else if piece.len() > 1 && piece.len() < 100 {
            _byte_pair_merge(&self.encoder, piece).len() - 1
        } else {
            byte_pair_encode(piece, &self.encoder).len()
        }
    }

    pub(crate) fn try_count_ranges_indexed(
        &self,
        text: &str,
        ranges: &[(usize, usize)],
        allowed: &HashSet<&str>,
    ) -> Option<Vec<usize>> {
        if self.regular_regex_tls.is_none()
            || text.is_empty()
            || text.len() > 65536
            || ranges.len() < 2
            // With no registrations the special regex is zero-width; an
            // unknown allowed empty string must keep the original panic path.
            || self.special_tokens_encoder.is_empty()
            || self.special_tokens_encoder.contains_key("")
            // A disallowed literal can hide an allowed prefix until EOF cuts
            // that literal. RMCP enables every registration; other policies
            // keep independent scalar counting and its exact search behavior.
            || !self.special_tokens_encoder.keys().all(|literal| allowed.contains(literal.as_str()))
        {
            return None;
        }
        // Invalid ranges retain the original scalar/error order. Counting any
        // unrequested text is safe only with a complete single-byte vocabulary.
        let work = ranges.iter().try_fold(0usize, |sum, &(start, end)| {
            text.get(start..end)?;
            sum.checked_add(end - start)
        })?;
        if work < 2 * text.len() || !(0..=255u8).all(|byte| self.encoder.contains_key(&[byte][..]))
        {
            return None;
        }
        let mut ends = Vec::new();
        let mut totals = vec![0usize];
        self.visit_count_pieces(text, allowed, |end, piece, special| {
            ends.push(end);
            totals.push(totals.last().unwrap() + self.count_index_piece(piece.as_bytes(), special));
            true
        })
        .ok()?;
        if ends.last().copied() != Some(text.len()) {
            return None;
        }
        ranges
            .iter()
            .map(|&(start, end)| {
                if start == end {
                    return Some(0);
                }
                // The pinned pattern is context-free except for its whitespace
                // lookahead. EOF can change the final piece and its predecessor.
                let last = ends.partition_point(|&boundary| boundary < end);
                let suffix = last.saturating_sub(1);
                let cut = if suffix == 0 { 0 } else { ends[suffix - 1] };
                if cut <= start {
                    return self.count_tokens(&text[start..end], allowed).ok();
                }
                let mut prefix = 0;
                let mut aligned = None;
                self.visit_count_pieces(&text[start..end], allowed, |boundary, piece, special| {
                    prefix += self.count_index_piece(piece.as_bytes(), special);
                    if start + boundary <= cut {
                        if let Ok(position) = ends.binary_search(&(start + boundary)) {
                            aligned = Some(position + 1);
                            return false;
                        }
                    }
                    true
                })
                .ok()?;
                match aligned {
                    // A matching boundary synchronizes the stateless matcher.
                    Some(position) => Some(
                        prefix + totals[suffix] - totals[position]
                            + self.count_tokens(&text[cut..end], allowed).ok()?,
                    ),
                    // No reusable interior: the visit already counted the range.
                    None => Some(prefix),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HashMap, Rank};

    fn core(pattern: &str) -> CoreBPE {
        let mut ranks: HashMap<Vec<u8>, Rank> =
            (0..=255u8).map(|byte| (vec![byte], byte as Rank)).collect();
        for (index, word) in [
            b"ab".as_slice(),
            b"abc",
            b"aa",
            b"  ",
            b"\r\n",
            "界".as_bytes(),
        ]
        .iter()
        .enumerate()
        {
            ranks.insert(word.to_vec(), 256 + index as Rank);
        }
        CoreBPE::new_internal(
            ranks,
            HashMap::from_iter([
                ("<S>".to_owned(), 400),
                ("<T>".to_owned(), 401),
                ("aba".to_owned(), 402),
                ("bab".to_owned(), 403),
                ("<S>long".to_owned(), 404),
                ("éS".to_owned(), 405),
            ]),
            pattern,
        )
        .unwrap()
    }

    fn check(model: &CoreBPE, text: &str, allowed: &HashSet<&str>) {
        let boundaries: Vec<_> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain([text.len()])
            .collect();
        let mut ranges: Vec<_> = boundaries
            .iter()
            .enumerate()
            .flat_map(|(i, &start)| boundaries[i..].iter().map(move |&end| (start, end)))
            .collect();
        ranges.extend([(0, text.len()); 2]);
        let expected: Vec<_> = ranges
            .iter()
            .map(|&(start, end)| {
                let span = &text[start..end];
                let ids = model.encode(span, allowed).unwrap().0;
                assert_eq!(model.count_tokens(span, allowed).unwrap(), ids.len());
                ids.len()
            })
            .collect();
        if text.is_empty()
            || !model
                .special_tokens_encoder
                .keys()
                .all(|literal| allowed.contains(literal.as_str()))
        {
            assert!(model
                .try_count_ranges_indexed(text, &ranges, allowed)
                .is_none());
        } else {
            assert_eq!(
                model
                    .try_count_ranges_indexed(text, &ranges, allowed)
                    .unwrap(),
                expected,
                "{text:?}"
            );
            ranges.reverse();
            assert_eq!(
                model
                    .try_count_ranges_indexed(text, &ranges, allowed)
                    .unwrap(),
                expected.into_iter().rev().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn every_short_range_preserves_eof_whitespace_and_contractions() {
        let model = core(include_str!("rmcp_pattern.txt"));
        let alphabet = ['a', 's', '\'', ' ', '\n', '!'];
        let allowed = model
            .special_tokens_encoder
            .keys()
            .map(String::as_str)
            .collect();
        for size in 0..=5u32 {
            for mut code in 0..alphabet.len().pow(size) {
                let mut text = String::new();
                for _ in 0..size {
                    text.push(alphabet[code % alphabet.len()]);
                    code /= alphabet.len();
                }
                check(&model, &text, &allowed);
            }
        }
    }

    #[test]
    fn every_unicode_special_and_overlapping_literal_boundary_matches_ids() {
        let model = core(include_str!("rmcp_pattern.txt"));
        for allowed in [
            HashSet::new(),
            HashSet::from(["<S>", "aba", "éS"]),
            HashSet::from(["<S>", "<T>", "aba", "bab", "<S>long", "éS"]),
        ] {
            for text in [
                "éS界😀\0 a\u{85}\u{2000}\r\n 're  ",
                "ababa ababababab <S>long<T>abc",
                "   aa\t\r\n \t a  'sfoo'rd!",
            ] {
                check(&model, text, &allowed);
            }
            for size in [2, 98, 99, 100, 101, 256] {
                let text = format!("界<S>{}<T>  éS", "a".repeat(size));
                let ranges = [(0, text.len()); 8];
                let expected = model.encode(&text, &allowed).unwrap().0.len();
                if model
                    .special_tokens_encoder
                    .keys()
                    .all(|literal| allowed.contains(literal.as_str()))
                {
                    assert_eq!(
                        model.try_count_ranges_indexed(&text, &ranges, &allowed),
                        Some(vec![expected; 8])
                    );
                } else {
                    assert!(model
                        .try_count_ranges_indexed(&text, &ranges, &allowed)
                        .is_none());
                }
            }
        }
    }

    #[test]
    fn unsupported_work_and_invalid_ranges_decline_before_speculation() {
        let model = core(include_str!("rmcp_pattern.txt"));
        let allowed = model
            .special_tokens_encoder
            .keys()
            .map(String::as_str)
            .collect();
        assert!(model
            .try_count_ranges_indexed("abc", &[(0, 3)], &allowed)
            .is_none());
        assert!(model
            .try_count_ranges_indexed("abc", &[(0, 1); 2], &allowed)
            .is_none());
        for invalid in [(0, 1), (1, 1), (2, 0), (0, 9)] {
            assert!(model
                .try_count_ranges_indexed("é", &[(0, 2), invalid], &allowed)
                .is_none());
        }
        let text = "x".repeat(65537);
        assert!(model
            .try_count_ranges_indexed(&text, &[(0, text.len()); 2], &allowed)
            .is_none());
        let incomplete = CoreBPE::new_internal(
            HashMap::from_iter([(b"a".to_vec(), 0)]),
            HashMap::from_iter([("<S>".to_owned(), 1)]),
            include_str!("rmcp_pattern.txt"),
        )
        .unwrap();
        assert!(incomplete
            .try_count_ranges_indexed("a!", &[(0, 1); 8], &allowed)
            .is_none());
        assert_eq!(incomplete.count_tokens("a", &allowed).unwrap(), 1);
        let arbitrary = core(r"[\s\S]+");
        assert!(arbitrary
            .try_count_ranges_indexed("abc", &[(0, 3); 8], &allowed)
            .is_none());
    }

    #[test]
    fn simultaneous_indexes_share_no_range_state() {
        let model = core(include_str!("rmcp_pattern.txt"));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let model = &model;
                scope.spawn(move || {
                    for _ in 0..10 {
                        let allowed = model
                            .special_tokens_encoder
                            .keys()
                            .map(String::as_str)
                            .collect();
                        check(model, "é<S>ababa  \r\n 's a!", &allowed);
                    }
                });
            }
        });
    }

    #[test]
    fn partial_special_policy_declines_even_when_full_text_hides_an_allowed_prefix() {
        let model = core(include_str!("rmcp_pattern.txt"));
        let text = "ababa ababababab <S>long<T>abc";
        let ranges = [(0, text.len()), (0, 20), (0, text.len())];
        for allowed in [HashSet::new(), HashSet::from(["<S>", "aba", "éS"])] {
            assert!(model
                .try_count_ranges_indexed(text, &ranges, &allowed)
                .is_none());
        }
    }

    #[test]
    fn empty_special_registry_keeps_original_counter() {
        let model = CoreBPE::new_internal(
            (0..=255u8).map(|byte| (vec![byte], byte as Rank)).collect(),
            HashMap::default(),
            include_str!("rmcp_pattern.txt"),
        ).unwrap();
        for allowed in [HashSet::new(), HashSet::from([""])] {
            assert!(model.try_count_ranges_indexed("abc", &[(0, 3); 2], &allowed).is_none());
        }
        assert_eq!(model.count_tokens("abc", &HashSet::new()).unwrap(), 3);
        let zero_width = CoreBPE::new_internal(
            (0..=255u8).map(|byte| (vec![byte], byte as Rank)).collect(),
            HashMap::from_iter([("".to_owned(), 256)]),
            include_str!("rmcp_pattern.txt"),
        ).unwrap();
        // Its original scalar API can loop. The index must decline before visiting it.
        assert!(zero_width.try_count_ranges_indexed("abc", &[(0, 3); 2], &HashSet::from([""])).is_none());
    }

}
