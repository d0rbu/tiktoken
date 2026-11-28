//! Exact source-interior reuse for the guarded RMCP pretokenizer.

use std::borrow::Cow;
use std::collections::HashSet;

use super::{CoreBPE, EncodeError, HashMap, Rank};

pub(crate) struct ReflectedEncoding {
    pub(crate) source: Vec<Rank>,
    pub(crate) physical: Vec<Rank>,
    pub(crate) source_ends: Vec<usize>,
    pub(crate) physical_ends: Vec<usize>,
}

#[derive(Clone, Copy)]
struct Interior<'a> {
    head: &'a str,
    first_anchor: usize,
    tail: &'a str,
}

fn ascii_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 11 | 12)
}

// The guarded pattern cannot match through this transition. Its optional
// leading space belongs to the right side, and the left side ends with a
// nonspace scalar, so neither whitespace run depends on the opposite side.
fn ordinary_cut(bytes: &[u8], position: usize) -> bool {
    position == 0
        || position == bytes.len()
        || ((b'!'..=b'~').contains(&bytes[position - 1]) && ascii_whitespace(bytes[position]))
}

// The caller advances through sorted character positions, so the start of
// each slice is already a character boundary.
fn char_point_byte(text: &str, point: usize) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    let (mut offset, mut remaining) = (0, point);
    #[cfg(not(target_arch = "x86_64"))]
    let (offset, remaining) = (0, point);
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8,
            _mm_set1_epi8,
        };

        let bytes = text.as_bytes();
        while remaining >= 16 && bytes.len() - offset >= 16 {
            // SSE2 is available on x86-64. The bounds check covers the entire load.
            let continuations = unsafe {
                let chunk = _mm_loadu_si128(bytes.as_ptr().add(offset).cast::<__m128i>());
                let tags = _mm_and_si128(chunk, _mm_set1_epi8(0xc0_u8 as i8));
                _mm_movemask_epi8(_mm_cmpeq_epi8(tags, _mm_set1_epi8(0x80_u8 as i8)))
            };
            remaining -= 16 - continuations.count_ones() as usize;
            offset += 16;
        }
        // The last character start may precede the last vector boundary.
        while bytes.get(offset).is_some_and(|byte| byte & 0xc0 == 0x80) {
            offset += 1;
        }
    }
    text[offset..]
        .char_indices()
        .map(|(index, _)| offset + index)
        .chain(std::iter::once(text.len()))
        .nth(remaining)
}

impl CoreBPE {
    fn append_reflection_fragment<'a>(
        &'a self,
        text: &'a str,
        allowed: &HashSet<&str>,
        parts: &mut Vec<(usize, Rank)>,
        reused: &mut HashMap<&'a [u8], (usize, usize)>,
        tokens: &mut Vec<Rank>,
        direct_byte: bool,
    ) -> Result<(), EncodeError> {
        // One byte of valid UTF-8 is one ASCII scalar. This guarded lexer
        // emits it as one ordinary piece unless an allowed special can match.
        if direct_byte
            && text.len() == 1
            && let Some(rank) = self.encoder.get_span(text.as_bytes(), 0, 1)
        {
            tokens.push(*rank);
            return Ok(());
        }
        self.encode_into_workspace::<false, _>(text, allowed, parts, reused, tokens, |_, _| {})?;
        Ok(())
    }

    pub(crate) fn encode_reflection_fragments(
        &self,
        rendered: &str,
        points: &[usize],
        source_spans: &[(usize, usize)],
        allowed: &HashSet<&str>,
    ) -> Result<Option<ReflectedEncoding>, EncodeError> {
        if self.regular_regex_tls.is_none()
            || allowed
                .iter()
                .any(|literal| literal.bytes().any(ascii_whitespace))
            || points.len() < 2
            || source_spans.is_empty()
            || points.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Ok(None);
        }
        let byte_points = if rendered.is_ascii() {
            if points.last().copied().unwrap_or(0) > rendered.len() {
                return Ok(None);
            }
            Cow::Borrowed(points)
        } else {
            let mut offset = 0;
            let mut previous_point = 0;
            let mut offsets = Vec::with_capacity(points.len());
            for &point in points {
                let Some(advance) = char_point_byte(&rendered[offset..], point - previous_point)
                else {
                    return Ok(None);
                };
                offset += advance;
                offsets.push(offset);
                previous_point = point;
            }
            Cow::Owned(offsets)
        };
        let mut which = vec![None; points.len() - 1];
        let mut source_ends = Vec::with_capacity(source_spans.len());
        let mut offset = 0;
        let mut physical = 0;
        for (index, &(start, end)) in source_spans.iter().enumerate() {
            while physical < which.len() && points[physical] < start {
                physical += 1;
            }
            if physical == which.len() || points[physical] != start || points[physical + 1] != end {
                return Ok(None);
            }
            offset += byte_points[physical + 1] - byte_points[physical];
            source_ends.push(offset);
            which[physical] = Some(index);
            physical += 1;
        }
        let source = if source_spans.len() == which.len() {
            Cow::Borrowed(&rendered[byte_points[0]..byte_points[which.len()]])
        } else {
            let mut joined = String::with_capacity(offset);
            for (piece, index) in byte_points.windows(2).zip(&which) {
                if index.is_some() {
                    joined.push_str(&rendered[piece[0]..piece[1]]);
                }
            }
            Cow::Owned(joined)
        };
        let text = source.as_ref();
        let bytes = text.as_bytes();
        let mut plans = Vec::with_capacity(source_ends.len());
        let mut anchors = Vec::new();
        let mut start = 0;
        for &end in &source_ends {
            let sharing = (start..end)
                .find(|&cut| ordinary_cut(bytes, cut))
                .and_then(|left| {
                    (left + 1..=end)
                        .rfind(|&cut| ordinary_cut(bytes, cut))
                        .map(|right| {
                            let key = anchors.len();
                            anchors.push(left);
                            anchors.push(right);
                            Interior {
                                head: &text[start..left],
                                first_anchor: key,
                                tail: &text[right..end],
                            }
                        })
                });
            plans.push(sharing);
            start = end;
        }
        let mut ranks = Vec::with_capacity(anchors.len());
        let mut tokens = Vec::new();
        let mut parts = Vec::new();
        let mut reused = HashMap::default();
        self.encode_into_workspace::<false, _>(
            text,
            allowed,
            &mut parts,
            &mut reused,
            &mut tokens,
            |byte, rank| {
                while anchors.get(ranks.len()) == Some(&byte) {
                    ranks.push(rank);
                }
            },
        )?;
        let source_count = tokens.len();
        while anchors.get(ranks.len()) == Some(&text.len()) {
            ranks.push(source_count);
        }
        if ranks.len() != anchors.len() {
            return Ok(None);
        }
        let direct_byte = allowed.iter().all(|special| special.len() > 1);
        let mut ends = Vec::with_capacity(which.len());
        for (bounds, index) in byte_points.windows(2).zip(which) {
            if let Some(Interior {
                head,
                first_anchor,
                tail,
            }) = index.and_then(|index| plans[index])
            {
                if !head.is_empty() {
                    self.append_reflection_fragment(
                        head,
                        allowed,
                        &mut parts,
                        &mut reused,
                        &mut tokens,
                        direct_byte,
                    )?;
                }
                tokens.extend_from_within(ranks[first_anchor]..ranks[first_anchor + 1]);
                if !tail.is_empty() {
                    self.append_reflection_fragment(
                        tail,
                        allowed,
                        &mut parts,
                        &mut reused,
                        &mut tokens,
                        direct_byte,
                    )?;
                }
            } else {
                self.append_reflection_fragment(
                    &rendered[bounds[0]..bounds[1]],
                    allowed,
                    &mut parts,
                    &mut reused,
                    &mut tokens,
                    direct_byte,
                )?;
            }
            ends.push(tokens.len() - source_count);
        }
        let physical = tokens.split_off(source_count);
        Ok(Some(ReflectedEncoding {
            source: tokens,
            physical,
            source_ends,
            physical_ends: ends,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_point_byte_matches_scalar_offsets_across_chunk_boundaries() {
        for padding in 0..32 {
            let text = format!(
                "{}{}",
                "a".repeat(padding),
                "é\u{0301}\0中🙂 alpha ".repeat(10)
            );
            let offsets: Vec<_> = text
                .char_indices()
                .map(|(offset, _)| offset)
                .chain(std::iter::once(text.len()))
                .collect();
            for (point, &offset) in offsets.iter().enumerate() {
                assert_eq!(char_point_byte(&text, point), Some(offset));
                assert_eq!(
                    char_point_byte(&text[offset..], offsets.len() - 1 - point),
                    Some(text.len() - offset)
                );
            }
            assert_eq!(char_point_byte(&text, offsets.len()), None);
        }
    }

    #[test]
    fn single_byte_fragments_preserve_allowed_specials_and_all_ascii_classes() {
        let vocab = (0u8..=255)
            .map(|byte| (vec![byte], Rank::from(byte)))
            .collect::<HashMap<_, _>>();
        let model = CoreBPE::new_internal(
            vocab, [("x".into(), 256), ("<r>".into(), 257)].into_iter().collect(),
            include_str!("rmcp_pattern.txt"),
        ).unwrap();
        let rendered: String = (0u8..=127).map(char::from).collect();
        let points: Vec<_> = (0..=128).collect();
        for allowed in [HashSet::from(["x"]), HashSet::from(["<r>"])] {
            let outcome = model
                .encode_reflection_fragments(&rendered, &points, &[(0, 1)], &allowed)
                .unwrap()
                .unwrap();
            let (physical, ends) = model.encode_batch_packed(
                rendered.as_bytes().chunks(1).map(|piece| std::str::from_utf8(piece).unwrap()),
                &allowed,
            ).unwrap();
            assert_eq!(outcome.source, [0]);
            assert_eq!(outcome.physical, physical);
            assert_eq!(outcome.physical_ends, ends);
        }
    }

    fn model(pattern: &str) -> CoreBPE {
        let mut vocab = (0u8..=255)
            .map(|byte| (vec![byte], Rank::from(byte)))
            .collect::<HashMap<_, _>>();
        vocab.insert(b" b".to_vec(), 256);
        vocab.insert(b"he".to_vec(), 257);
        CoreBPE::new_internal(
            vocab,
            [
                ("<r>".to_owned(), 258),
                ("</r>".to_owned(), 259),
                (" <r>".to_owned(), 260),
            ]
            .into_iter()
            .collect(),
            pattern,
        )
        .unwrap()
    }

    #[test]
    fn reuses_source_interiors_with_unicode_and_independent_feedback() {
        let encoding = model(include_str!("rmcp_pattern.txt"));
        let pieces = ["a hello ", "<r>雪</r>", "b here now"];
        let rendered = pieces.concat();
        let original = [pieces[0], pieces[2]].concat();
        let points = [0, 8, 16, 26];
        assert_eq!(rendered.chars().count(), points[3]);
        let spans = [(points[0], points[1]), (points[2], points[3])];
        let allowed = HashSet::from(["<r>", "</r>"]);
        let outcome = encoding
            .encode_reflection_fragments(&rendered, &points, &spans, &allowed)
            .unwrap()
            .unwrap();
        let (expected, ends) = encoding
            .encode_batch_packed(pieces.into_iter(), &allowed)
            .unwrap();
        let (source, _) = encoding.encode(&original, &allowed).unwrap();
        assert_eq!(outcome.source, source);
        assert_eq!(outcome.physical, expected);
        assert_eq!(outcome.physical_ends, ends);
        assert_eq!(outcome.source_ends, [pieces[0].len(), original.len()]);
    }

    #[test]
    fn declines_unrecognized_patterns_and_nonexact_source_fragments() {
        let generic = model(r"[\s\S]+");
        let guarded = model(include_str!("rmcp_pattern.txt"));
        let allowed = HashSet::from(["<r>"]);
        assert!(
            generic
                .encode_reflection_fragments("a b", &[0, 3], &[(0, 3)], &allowed)
                .unwrap()
                .is_none()
        );
        assert!(
            guarded
                .encode_reflection_fragments("a b", &[0, 1, 3], &[(0, 3)], &allowed)
                .unwrap()
                .is_none()
        );
        assert!(
            guarded
                .encode_reflection_fragments("a b", &[0, 3], &[(0, 3)], &HashSet::from([" <r>"]))
                .unwrap()
                .is_none()
        );
    }
}
