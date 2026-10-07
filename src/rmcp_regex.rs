//! Isolated prototype for one byte-exact, HF-derived RMCP pattern.
//!
//! All other patterns keep the original fancy-regex iterator. The admitted
//! pattern has no contextual assertions except its penultimate whitespace
//! branch. Its explicit Unicode tables are retained verbatim.

const GUARDED_PATTERN: &str = include_str!("rmcp_pattern.txt");
// Exact table in the captured HF pattern, not Rust's Unicode whitespace API.
fn is_guarded_whitespace(scalar: char) -> bool {
    matches!(
        scalar,
        '\u{9}'..='\u{d}' | '\u{20}' | '\u{85}' | '\u{a0}' | '\u{1680}'
            | '\u{2000}'..='\u{200a}' | '\u{2028}'..='\u{2029}' | '\u{202f}'
            | '\u{205f}' | '\u{3000}'
    )
}

pub(crate) fn compile_if_guarded(pattern: &str) -> Result<Option<regex::Regex>, regex::Error> {
    if pattern != GUARDED_PATTERN {
        return Ok(None);
    }

    let invalid = || regex::Error::Syntax("unexpected guarded RMCP whitespace suffix".to_owned());
    let (before_last, whitespace) = pattern.rsplit_once('|').ok_or_else(invalid)?;
    let (prefix, lookahead) = before_last.rsplit_once('|').ok_or_else(invalid)?;
    let class = whitespace
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix("]+"))
        .ok_or_else(invalid)?;
    if lookahead != format!("{whitespace}(?![^{class}])") {
        return Err(invalid());
    }
    regex::Regex::new(&format!("{prefix}|{whitespace}")).map(Some)
}

pub(crate) enum Pieces<'r, 't, I> {
    Original(I),
    Regular(RegularPieces<'r, 't>),
}

impl<'t, I> Iterator for Pieces<'_, 't, I>
where
    I: Iterator<Item = Result<&'t str, fancy_regex::Error>>,
{
    type Item = Result<&'t str, fancy_regex::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Original(original) => original.next(),
            Self::Regular(regular) => regular.next().map(Ok),
        }
    }
}

pub(crate) struct RegularPieces<'r, 't> {
    regex: &'r regex::Regex,
    text: &'t str,
    base: usize,
    matches: regex::Matches<'r, 't>,
}

impl<'r, 't> RegularPieces<'r, 't> {
    pub(crate) fn new(regex: &'r regex::Regex, text: &'t str) -> Self {
        Self {
            regex,
            text,
            base: 0,
            matches: regex.find_iter(text),
        }
    }
}

impl<'t> Iterator for RegularPieces<'_, 't> {
    type Item = &'t str;

    fn next(&mut self) -> Option<Self::Item> {
        let matched = self.matches.next()?;
        let start = self.base + matched.start();
        let mut end = self.base + matched.end();

        // At the end of this ordinary segment, the original negative lookahead
        // succeeds, even when an allowed special follows in the full input.
        // Otherwise its greedy whitespace branch withholds the final scalar
        // only when at least two whitespace scalars are available. A single
        // scalar is consumed by the original final whitespace alternative.
        // Only the whitespace alternative can end in this exact table: the
        // earlier L/N/other alternatives end outside it, as do contractions.
        if end < self.text.len() {
            let (final_offset, final_scalar) = matched.as_str()
                .char_indices()
                .next_back()
                .expect("the guarded alternatives are nonempty");
            if is_guarded_whitespace(final_scalar) && final_offset > 0 {
                end = start + final_offset;
                // Restart ALL alternatives at the withheld scalar. Attaching it
                // to the next existing match is wrong before contractions.
                // Slice restart is valid only for this context-free pattern.
                self.base = end;
                self.matches = self.regex.find_iter(&self.text[end..]);
            }
        }
        Some(&self.text[start..end])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{GUARDED_PATTERN, RegularPieces, compile_if_guarded, is_guarded_whitespace};
    use crate::{CoreBPE, HashMap, Rank};

    const GUARDED_WHITESPACE: &str = r"\x{9}-\x{d}\x{20}\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}-\x{2029}\x{202f}\x{205f}\x{3000}";

    fn matchers() -> (fancy_regex::Regex, regex::Regex) {
        (
            fancy_regex::Regex::new(GUARDED_PATTERN).unwrap(),
            compile_if_guarded(GUARDED_PATTERN).unwrap().unwrap(),
        )
    }

    fn assert_spans_equal(original: &fancy_regex::Regex, regular: &regex::Regex, text: &str) {
        let expected: Vec<_> = original
            .find_iter(text)
            .map(|matched| {
                let matched = matched.unwrap();
                (matched.start(), matched.end())
            })
            .collect();
        let actual: Vec<_> = RegularPieces::new(regular, text)
            .map(|piece| {
                let start = piece.as_ptr() as usize - text.as_ptr() as usize;
                assert!(!piece.is_empty());
                (start, start + piece.len())
            })
            .collect();
        assert_eq!(actual, expected, "input {text:?}");
        assert_eq!(actual.first().map_or(0, |span| span.0), 0);
        assert_eq!(actual.last().map_or(0, |span| span.1), text.len());
        assert!(actual.windows(2).all(|pair| pair[0].1 == pair[1].0));
    }

    fn core(pattern: &str) -> CoreBPE {
        let mut ranks: HashMap<Vec<u8>, Rank> =
            (0_u16..256).map(|byte| (vec![byte as u8], byte as Rank)).collect();
        // These merges expose changed whitespace/contraction piece boundaries.
        for (offset, bytes) in [b" '".as_slice(), b"'s", b"  ", b" a"].iter().enumerate() {
            ranks.insert(bytes.to_vec(), 256 + offset as Rank);
        }
        let specials = HashMap::from_iter([("<S>".to_owned(), 300), ("<T>".to_owned(), 301)]);
        CoreBPE::new_internal(ranks, specials, pattern).unwrap()
    }

    #[test]
    fn exact_pattern_activates_and_near_matches_fall_back() {
        assert!(compile_if_guarded(GUARDED_PATTERN).unwrap().is_some());
        let equivalent = format!("(?:{GUARDED_PATTERN})");
        for pattern in [equivalent.as_str(), "[a-z]+", ""] {
            assert!(compile_if_guarded(pattern).unwrap().is_none());
            assert!(core(pattern).regular_regex_tls.is_none());
        }
        assert!(core(GUARDED_PATTERN).regular_regex_tls.is_some());
        // A malformed non-guarded pattern still fails original construction.
        assert!(CoreBPE::new_internal(HashMap::default(), HashMap::default(), "[").is_err());
    }

    #[test]
    fn exact_whitespace_classifier_matches_every_unicode_scalar() {
        let whitespace = GUARDED_PATTERN.rsplit_once('|').unwrap().1;
        let class = whitespace.strip_prefix('[').unwrap().strip_suffix("]+").unwrap();
        assert_eq!(class, GUARDED_WHITESPACE);
        let exact_class = regex::Regex::new(&format!(r"\A[{class}]\z")).unwrap();
        let mut count = 0;
        let mut whitespace_count = 0;
        for scalar in (0..=0x10ffff).filter_map(char::from_u32) {
            let mut buffer = [0; 4];
            let text = scalar.encode_utf8(&mut buffer);
            let classified = is_guarded_whitespace(scalar);
            assert_eq!(classified, exact_class.is_match(text), "scalar {scalar:?}");
            count += 1;
            whitespace_count += usize::from(classified);
        }
        assert_eq!(count, 1_112_064);
        assert_eq!(whitespace_count, 25);
    }

    #[test]
    fn last_scalar_identifies_whitespace_without_capture_groups() {
        let (original, regular) = matchers();
        // Group zero only: no named or unnamed captures are compiled.
        assert_eq!(regular.captures_len(), 1);
        for scalar in [
            '\u{9}', '\u{a}', '\u{b}', '\u{c}', '\u{d}', ' ', '\u{85}', '\u{a0}', '\u{1680}',
            '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
            '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}',
            '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}', '\u{1c}', '\u{1f}',
            '\u{200b}', '\u{feff}', 'é', '३', '\u{301}', '😀', '\'', '!',
        ] {
            for suffix in ["", "a", "1", "!", "'s", "'re", "'S", "<S>"] {
                let text = format!("a{scalar} {scalar}{suffix}");
                for matched in regular.find_iter(&text) {
                    let piece = matched.as_str();
                    let last = piece.chars().next_back().unwrap();
                    assert_eq!(is_guarded_whitespace(last), piece.chars().all(is_guarded_whitespace), "piece {piece:?}");
                }
                assert_spans_equal(&original, &regular, &text);
            }
        }
    }

    #[test]
    fn contractions_restart_at_withheld_scalar() {
        let (original, regular) = matchers();
        for text in [
            "  's", "\t 's", " \t's", "\u{a0} 's", "  're", "  'll", "  'S", "a  'd!",
            "  't", "  've", "  'm", "   '!", " 's", "\t's",
        ] {
            assert_spans_equal(&original, &regular, text);
        }
        assert_eq!(RegularPieces::new(&regular, "  's").collect::<Vec<_>>(), [" ", " '", "s"]);
        assert_eq!(RegularPieces::new(&regular, "\t 's").collect::<Vec<_>>(), ["\t", " '", "s"]);
    }

    #[test]
    fn eof_and_single_scalar_whitespace_make_progress() {
        let (original, regular) = matchers();
        for text in [
            "", " ", "  ", "\t", "\n\t ", "\u{a0}", "\u{2003}\u{3000}", "a \t\u{a0}",
            "\t!", "\u{2003}a", " \u{2003}a", "\u{2003} a", "\r\n!",
        ] {
            assert_spans_equal(&original, &regular, text);
        }
        assert_eq!(RegularPieces::new(&regular, " \t\u{a0}").collect::<Vec<_>>(), [" \t\u{a0}"]);
    }

    #[test]
    fn explicit_unicode_classes_and_utf8_boundaries_match() {
        let (original, regular) = matchers();
        for text in [
            "élève 中文 １２३", "a\u{301} 𝟠!😀", "\u{85}\u{a0}'s", "\u{1680}\u{2003}x",
            "\u{2028}\u{2029}'re", "\u{202f}\u{205f}7", "\u{3000} 😀", "\u{200b}\u{feff}x",
            "\u{0}a\u{10ffff}", "\u{1c}\u{1f} a", "\u{e000}\u{fdd0}",
        ] {
            assert_spans_equal(&original, &regular, text);
        }
    }

    #[test]
    fn allowed_special_boundary_is_an_independent_regex_eof() {
        let (original, regular) = matchers();
        let text = "x \t\u{a0}<S>  's<T><S>\u{2003} ";
        for segment in text.split("<S>").flat_map(|part| part.split("<T>")) {
            assert_spans_equal(&original, &regular, segment);
        }
        assert_eq!(RegularPieces::new(&regular, "x \t\u{a0}").collect::<Vec<_>>(), ["x", " \t\u{a0}"]);
    }

    #[test]
    fn bounded_exhaustive_actual_pattern_spans_match() {
        fn visit(original: &fancy_regex::Regex, regular: &regex::Regex, text: &mut String, remaining: usize) -> usize {
            assert_spans_equal(original, regular, text);
            let mut count = 1;
            if remaining > 0 {
                for scalar in [' ', '\t', '\n', '\u{a0}', '\'', 's', 'a', '1', '!'] {
                    let previous = text.len();
                    text.push(scalar);
                    count += visit(original, regular, text, remaining - 1);
                    text.truncate(previous);
                }
            }
            count
        }
        let (original, regular) = matchers();
        assert_eq!(visit(&original, &regular, &mut String::new(), 5), 66_430);
    }

    #[test]
    fn both_encode_routes_match_forced_original_with_specials() {
        let candidate = core(GUARDED_PATTERN);
        let original = core(&format!("(?:{GUARDED_PATTERN})"));
        assert!(candidate.regular_regex_tls.is_some());
        assert!(original.regular_regex_tls.is_none());
        let allowed_sets = [HashSet::new(), HashSet::from(["<S>"]), HashSet::from(["<S>", "<T>"])];
        for text in [
            "", "  's", "\t 's", "a  ", "é \u{3000}'re😀", "  <S>'s", "\t <S>  's<T> ",
            "<S><T>", "<S> \u{2003}<T>", "  <S>  ", "x<T> <S>'s<T>",
        ] {
            assert_eq!(candidate.encode_ordinary(text), original.encode_ordinary(text), "{text:?}");
            for allowed in &allowed_sets {
                assert_eq!(candidate.encode(text, allowed).unwrap(), original.encode(text, allowed).unwrap(), "{text:?}, {allowed:?}");
            }
        }
    }
}
