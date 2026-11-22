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

impl CoreBPE {
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
            let mut indices = rendered.char_indices().chain([(rendered.len(), '\0')]);
            let mut next = 0;
            let mut offsets = Vec::with_capacity(points.len());
            for &point in points {
                let Some((offset, _)) = indices.nth(point - next) else {
                    return Ok(None);
                };
                offsets.push(offset);
                next = point + 1;
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
        let mut ends = Vec::with_capacity(which.len());
        for (bounds, index) in byte_points.windows(2).zip(which) {
            if let Some(Interior {
                head,
                first_anchor,
                tail,
            }) = index.and_then(|index| plans[index])
            {
                if !head.is_empty() {
                    self.encode_into_workspace::<false, _>(
                        head,
                        allowed,
                        &mut parts,
                        &mut reused,
                        &mut tokens,
                        |_, _| {},
                    )?;
                }
                tokens.extend_from_within(ranks[first_anchor]..ranks[first_anchor + 1]);
                if !tail.is_empty() {
                    self.encode_into_workspace::<false, _>(
                        tail,
                        allowed,
                        &mut parts,
                        &mut reused,
                        &mut tokens,
                        |_, _| {},
                    )?;
                }
            } else {
                self.encode_into_workspace::<false, _>(
                    &rendered[bounds[0]..bounds[1]],
                    allowed,
                    &mut parts,
                    &mut reused,
                    &mut tokens,
                    |_, _| {},
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
