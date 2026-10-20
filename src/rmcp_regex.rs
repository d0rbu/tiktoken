//! Isolated prototype for one byte-exact, HF-derived RMCP pattern.
//!
//! All other patterns keep the original fancy-regex iterator. The admitted
//! pattern has no contextual assertions except its penultimate whitespace
//! branch. Its explicit Unicode tables are retained verbatim.

use std::error::Error;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::Arc;

use regex_automata::{
    Anchored, Input, MatchError, MatchKind,
    hybrid::dfa,
    nfa::thompson,
    util::{pool::{Pool, PoolGuard}, syntax},
};

type CacheFactory = Box<dyn Fn() -> dfa::Cache + Send + Sync + UnwindSafe + RefUnwindSafe>;

#[derive(Debug)]
pub(crate) struct RegularRegex {
    dfa: Arc<dfa::DFA>,
    caches: Pool<dfa::Cache, CacheFactory>,
}

impl RegularRegex {
    fn from_dfa(dfa: Arc<dfa::DFA>) -> Self {
        let shared = Arc::clone(&dfa);
        let create: CacheFactory = Box::new(move || shared.create_cache());
        Self { dfa, caches: Pool::new(create) }
    }
}

impl Clone for RegularRegex {
    fn clone(&self) -> Self {
        // As with meta::Regex, clones share the compiled automaton but own
        // separate cache pools. Nested/concurrent iterators hold separate guards.
        Self::from_dfa(Arc::clone(&self.dfa))
    }
}

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

// Keep the original parser/NFA limits and public syntax/size error conversion.
// Only the exact guarded pattern reaches this builder in production. It has no
// assertions needing a reverse search, and every next match start is known.
fn build_regular_regex(pattern: &str) -> Result<RegularRegex, regex::Error> {
    let nfa = thompson::NFA::compiler()
        .configure(
            thompson::Config::new()
                .nfa_size_limit(Some(10 * (1 << 20)))
                .which_captures(thompson::WhichCaptures::None)
                .utf8(true),
        )
        .syntax(syntax::Config::default().utf8(true))
        .build(pattern)
        .map_err(|error| {
            if let Some(size_limit) = error.size_limit() {
                regex::Error::CompiledTooBig(size_limit)
            } else if let Some(source) = error.source() {
                regex::Error::Syntax(source.to_string())
            } else {
                regex::Error::Syntax(error.to_string())
            }
        })?;
    let dfa = dfa::DFA::builder()
        .configure(
            dfa::DFA::config()
                .match_kind(MatchKind::LeftmostFirst)
                .cache_capacity(2 * (1 << 20)),
        )
        .build_from_nfa(nfa)
        .map_err(|error| regex::Error::Syntax(error.to_string()))?;
    Ok(RegularRegex::from_dfa(Arc::new(dfa)))
}

pub(crate) fn compile_if_guarded(pattern: &str) -> Result<Option<RegularRegex>, regex::Error> {
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
    build_regular_regex(&format!("{prefix}|{whitespace}")).map(Some)
}

/// Single-pattern anchored longest continuation. The DFA retains leftmost-first
/// alternative priority. A match is delayed by one byte; EOI is a real transition.
/// Public next_state resolves unknown states and preserves cache-owned state IDs.
/// The exact guarded builder uses no start specialization, prefilter, or cache
/// give-up limits. Slice restarts are valid only for this context-free pattern.
#[inline(never)]
fn match_state_search(
    dfa: &dfa::DFA,
    cache: &mut dfa::Cache,
    text: &str,
) -> Result<Option<usize>, MatchError> {
    let input = Input::new(text).anchored(Anchored::Yes);
    let mut sid = dfa.start_state_forward(cache, &input)?;
    let mut matched = usize::MAX;
    cache.search_start(0);
    for (at, byte) in text.bytes().enumerate() {
        cache.search_update(at);
        sid = dfa
            .next_state(cache, sid, byte)
            .map_err(|_| MatchError::gave_up(at))?;
        // Keep the greedy offset update as a conditional select in this hot loop.
        matched = if sid.is_match() { at } else { matched };
        if sid.is_dead() {
            cache.search_finish(at);
            return Ok((matched != usize::MAX).then_some(matched));
        }
        if sid.is_quit() {
            cache.search_finish(at);
            return Err(MatchError::quit(byte, at));
        }
    }
    cache.search_update(text.len());
    sid = dfa
        .next_eoi_state(cache, sid)
        .map_err(|_| MatchError::gave_up(text.len()))?;
    matched = if sid.is_match() { text.len() } else { matched };
    cache.search_finish(text.len());
    Ok((matched != usize::MAX).then_some(matched))
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
    regex: &'r RegularRegex,
    text: &'t str,
    input: Input<'t>,
    cache: PoolGuard<'r, dfa::Cache, CacheFactory>,
}

impl<'r, 't> RegularPieces<'r, 't> {
    pub(crate) fn new(regex: &'r RegularRegex, text: &'t str) -> Self {
        Self {
            regex,
            text,
            input: Input::new(text).anchored(Anchored::Yes),
            cache: regex.caches.get(),
        }
    }
}

impl<'t> Iterator for RegularPieces<'_, 't> {
    type Item = &'t str;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.input.start();
        if start == self.text.len() {
            return None;
        }
        // No quit bytes, word boundaries, or cache give-up limits are enabled;
        // this exact nonempty pattern covers each scalar at the anchored start.
        let matched = match_state_search(&self.regex.dfa, &mut self.cache, &self.text[start..])
            .expect("guarded forward DFA cannot quit")?;
        let mut end = start + matched;

        // EOF belongs to this ordinary segment even if an allowed special
        // follows. Otherwise withhold the final whitespace scalar exactly as
        // before, then restart ALL alternatives there (including contractions).
        if end < self.text.len() {
            let (final_offset, final_scalar) = self.text[start..end]
                .char_indices()
                .next_back()
                .expect("the guarded alternatives are nonempty");
            if is_guarded_whitespace(final_scalar) && final_offset > 0 {
                end = start + final_offset;
            }
        }
        self.input.set_start(end);
        Some(&self.text[start..end])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use regex_automata::Anchored;

    use super::{GUARDED_PATTERN, RegularRegex, RegularPieces, build_regular_regex, compile_if_guarded, is_guarded_whitespace};
    use crate::{CoreBPE, HashMap, Rank};

    const GUARDED_WHITESPACE: &str = r"\x{9}-\x{d}\x{20}\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}-\x{2029}\x{202f}\x{205f}\x{3000}";

    fn matchers() -> (fancy_regex::Regex, RegularRegex) {
        (
            fancy_regex::Regex::new(GUARDED_PATTERN).unwrap(),
            compile_if_guarded(GUARDED_PATTERN).unwrap().unwrap(),
        )
    }

    fn assert_spans_equal(original: &fancy_regex::Regex, regular: &RegularRegex, text: &str) {
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
        // The forward DFA needs no capture slots, including group zero.
        assert_eq!(regular.dfa.get_nfa().group_info().all_group_len(), 0);
        for scalar in [
            '\u{9}', '\u{a}', '\u{b}', '\u{c}', '\u{d}', ' ', '\u{85}', '\u{a0}', '\u{1680}',
            '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
            '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}',
            '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}', '\u{1c}', '\u{1f}',
            '\u{200b}', '\u{feff}', 'é', '३', '\u{301}', '😀', '\'', '!',
        ] {
            for suffix in ["", "a", "1", "!", "'s", "'re", "'S", "<S>"] {
                let text = format!("a{scalar} {scalar}{suffix}");
                for piece in RegularPieces::new(&regular, &text) {
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
        fn visit(original: &fancy_regex::Regex, regular: &RegularRegex, text: &mut String, remaining: usize) -> usize {
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

    #[test]
    fn anchored_iterator_reanchors_after_matches_and_whitespace_restarts() {
        let (original, regular) = matchers();
        let text = "aa11!!bb  's\u{3000}\u{a0}é  ";
        let mut pieces = RegularPieces::new(&regular, text);
        let mut actual = Vec::new();
        let mut saw_restart = false;
        loop {
            assert_eq!(pieces.input.get_anchored(), Anchored::Yes);
            assert!(!pieces.input.get_earliest());
            let Some(piece) = pieces.next() else { break };
            saw_restart |= piece == " " && text[pieces.input.start()..].starts_with(" ");
            actual.push(piece);
        }
        assert!(saw_restart);
        assert!(actual.len() > 4);
        assert_eq!(actual.concat(), text);
        assert_spans_equal(&original, &regular, text);

        // An unanchored iterator would skip the gap and return a second "a".
        // The admitted pattern has no gaps; this isolates anchoring itself.
        let simple = build_regular_regex("a").unwrap();
        assert_eq!(RegularPieces::new(&simple, "a a").collect::<Vec<_>>(), ["a"]);
    }

    #[test]
    fn regular_builder_preserves_public_syntax_and_size_errors() {
        for pattern in ["[", "a{1000000000}"] {
            let expected = regex::Regex::new(pattern).unwrap_err();
            let actual = build_regular_regex(pattern).unwrap_err();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn forward_caches_survive_nested_early_close_and_concurrent_use() {
        let (_, regular) = matchers();
        let regular = Arc::new(regular);
        let texts = ["", "aa11!!bb  's\u{3000}\u{a0}é  ", "\t 's", "é中文😀\0", " a  "];
        let original = fancy_regex::Regex::new(GUARDED_PATTERN).unwrap();
        let expected: Vec<Vec<String>> = texts.iter().map(|text| original.find_iter(text)
            .map(|m| m.unwrap().as_str().to_owned()).collect()).collect();
        let clone = regular.as_ref().clone();
        assert!(Arc::ptr_eq(&regular.dfa, &clone.dfa));
        // Keep one pooled cache checked out while another iterator comes and goes.
        let mut held = RegularPieces::new(&regular, texts[1]);
        assert_eq!(held.next(), Some("aa"));
        let mut early = RegularPieces::new(&regular, texts[2]);
        assert_eq!(early.next(), Some("\t"));
        drop(early);
        for (text, wanted) in texts.iter().zip(&expected) {
            assert_eq!(RegularPieces::new(&clone, text).collect::<Vec<_>>(), *wanted);
        }
        let handles: Vec<_> = (0..16).map(|_| {
            let regular = Arc::clone(&regular);
            let expected = expected.clone();
            std::thread::spawn(move || {
                for _ in 0..64 {
                    for (text, wanted) in texts.iter().zip(&expected) {
                        assert_eq!(RegularPieces::new(&regular, text).collect::<Vec<_>>(), *wanted);
                    }
                }
            })
        }).collect();
        for handle in handles { handle.join().unwrap(); }
        let tail: String = held.collect();
        assert_eq!(format!("aa{tail}"), texts[1]);
        let joined = texts.join("");
        assert_spans_equal(&original, &regular, &joined);
    }

    #[test]
    fn direct_forward_matches_every_unicode_scalar_with_surrounding_context() {
        let (original, regular) = matchers();
        let mut count = 0;
        for scalar in (0..=0x10ffff).filter_map(char::from_u32) {
            let text = format!("a{scalar}  's");
            assert_spans_equal(&original, &regular, &text);
            count += 1;
        }
        assert_eq!(count, 1_112_064);
    }

    #[test]
    fn match_state_search_preserves_priority_empty_input_and_error_offsets() {
        use regex_automata::{Input, hybrid::dfa};

        for pattern in [
            "a", "a|ab", "ab|a", "a+", "a+?", "a$", "a.*b", "a.*?b", "[a-z]+",
        ] {
            let dfa = dfa::DFA::new(pattern).unwrap();
            let mut original = dfa.create_cache();
            let mut candidate = dfa.create_cache();
            for text in ["", "a", "ab", "aaab!", "axbxby", " a", "a\na", "aé"] {
                let expected = dfa
                    .try_search_fwd(&mut original, &Input::new(text).anchored(Anchored::Yes))
                    .map(|m| m.map(|m| m.offset()));
                let actual = super::match_state_search(&dfa, &mut candidate, text);
                assert_eq!(
                    format!("{expected:?}"),
                    format!("{actual:?}"),
                    "{pattern} {text:?}"
                );
            }
        }
        let dfa = dfa::DFA::builder()
            .configure(dfa::DFA::config().quit(b'!', true))
            .build("a+")
            .unwrap();
        for text in ["!a", "a!", "aa", ""] {
            let expected = dfa
                .try_search_fwd(
                    &mut dfa.create_cache(),
                    &Input::new(text).anchored(Anchored::Yes),
                )
                .map(|m| m.map(|m| m.offset()));
            let actual = super::match_state_search(&dfa, &mut dfa.create_cache(), text);
            assert_eq!(
                format!("{expected:?}"),
                format!("{actual:?}"),
                "quit {text:?}"
            );
        }
    }


}
