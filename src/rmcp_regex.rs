//! Isolated prototype for one byte-exact, HF-derived RMCP pattern.
//!
//! All other patterns keep the original fancy-regex iterator. The admitted
//! pattern has no contextual assertions except its penultimate whitespace
//! branch. Its explicit Unicode tables are retained verbatim.

use std::error::Error;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::Arc;

use crate::Rank;
use crate::rmcp_ranks::EncoderRanks;

use regex_automata::{
    Anchored, Input, MatchError, MatchKind,
    hybrid::dfa,
    nfa::thompson,
    util::{
        pool::{Pool, PoolGuard},
        syntax,
    },
};

type CacheFactory = Box<dyn Fn() -> dfa::Cache + Send + Sync + UnwindSafe + RefUnwindSafe>;

#[derive(Debug)]
pub(crate) struct RegularRegex {
    dfa: Arc<dfa::DFA>,
    caches: Pool<dfa::Cache, CacheFactory>,
    ascii_guarded: bool,
}

impl RegularRegex {
    fn from_dfa(dfa: Arc<dfa::DFA>) -> Self {
        let shared = Arc::clone(&dfa);
        let create: CacheFactory = Box::new(move || shared.create_cache());
        Self {
            dfa,
            caches: Pool::new(create),
            ascii_guarded: false,
        }
    }
}

impl Clone for RegularRegex {
    fn clone(&self) -> Self {
        // As with meta::Regex, clones share the compiled automaton but own
        // separate cache pools. Nested/concurrent iterators hold separate guards.
        let mut cloned = Self::from_dfa(Arc::clone(&self.dfa));
        cloned.ascii_guarded = self.ascii_guarded;
        cloned
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
    let mut regex = build_regular_regex(&format!("{prefix}|{whitespace}"))?;
    regex.ascii_guarded = true;
    Ok(Some(regex))
}

fn ascii_class(byte: u8) -> u8 {
    match byte {
        b'A'..=b'Z' | b'a'..=b'z' => 1,
        b'0'..=b'9' => 2,
        9..=13 | b' ' => 0,
        0..=127 => 3,
        _ => 4,
    }
}

/// The exact guarded alternatives on ASCII. A non-ASCII boundary delegates
/// the entire piece, since its Unicode class can extend the current run.
fn ascii_piece_len(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes[0] == b'\'' {
        match bytes.get(1) {
            Some(b's' | b't' | b'm' | b'd') => return Some(2),
            Some(b'r' | b'v') if bytes.get(2) == Some(&b'e') => return Some(3),
            Some(b'l') if bytes.get(2) == Some(&b'l') => return Some(3),
            _ => {}
        }
    }
    let begin = usize::from(bytes[0] == b' ' && bytes.len() > 1);
    let class = ascii_class(bytes[begin]);
    if class == 4 {
        return None;
    }
    let mut end = begin + 1;
    while end < bytes.len() {
        let next = ascii_class(bytes[end]);
        if next == 4 {
            return None;
        }
        if next != class {
            break;
        }
        end += 1;
    }
    // EOF means the end of this ordinary segment, including before a special.
    if class == 0 && end < bytes.len() && end > 1 {
        end -= 1;
    }
    Some(end)
}

/// Mark complete ASCII piece ends as bit (end - 1), leaving any final run
/// that might continue beyond this block to the original scanner.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ascii_block_ends(text: &str) -> u64 {
    use std::arch::x86_64::*;

    let bytes = text.as_bytes();
    // The caller checks the length and CPU feature before entering this path.
    let mut letters = 0_u64;
    let mut digits = 0_u64;
    let mut whitespace = 0_u64;
    let mut spaces = 0_u64;
    let mut quotes = 0_u64;
    let mut high = 0_u64;
    for offset in [0, 32] {
        let v = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(offset).cast()) };
        let lower = _mm256_or_si256(v, _mm256_set1_epi8(32));
        let l = _mm256_and_si256(
            _mm256_cmpgt_epi8(lower, _mm256_set1_epi8(96)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(123), lower),
        );
        let n = _mm256_and_si256(
            _mm256_cmpgt_epi8(v, _mm256_set1_epi8(47)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(58), v),
        );
        let w = _mm256_or_si256(
            _mm256_cmpeq_epi8(v, _mm256_set1_epi8(32)),
            _mm256_and_si256(
                _mm256_cmpgt_epi8(v, _mm256_set1_epi8(8)),
                _mm256_cmpgt_epi8(_mm256_set1_epi8(14), v),
            ),
        );
        letters |= (_mm256_movemask_epi8(l) as u32 as u64) << offset;
        digits |= (_mm256_movemask_epi8(n) as u32 as u64) << offset;
        whitespace |= (_mm256_movemask_epi8(w) as u32 as u64) << offset;
        spaces |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(32))) as u32 as u64)
            << offset;
        quotes |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(39))) as u32 as u64)
            << offset;
        high |= (_mm256_movemask_epi8(v) as u32 as u64) << offset;
    }
    let limit = high.trailing_zeros() as usize;
    if limit == 0 {
        return 0;
    }
    let valid = u64::MAX >> (64 - limit);
    // Class changes are both the old run's end and the next run's start.
    let transitions =
        (letters ^ (letters << 1)) | (digits ^ (digits << 1)) | (whitespace ^ (whitespace << 1));
    // An ASCII space joins the following non-whitespace run. The whitespace
    // give-back is applied afterward so consecutive spaces keep their split.
    // Exclude the final classified byte: the following Unicode/block byte
    // can still extend that run.
    let give_back = whitespace & !(whitespace >> 1) & (valid >> 1);
    let mut cuts = ((transitions & !(spaces << 1)) | give_back) & valid;
    let mut contractions = quotes & (cuts | 1) & valid;
    let mut end64 = bytes.len() == 64 && limit == 64;
    while contractions != 0 {
        let at = contractions.trailing_zeros() as usize;
        contractions &= contractions - 1;
        let length = match bytes.get(at + 1) {
            Some(b's' | b't' | b'm' | b'd') => 2,
            Some(b'r' | b'v') if bytes.get(at + 2) == Some(&b'e') => 3,
            Some(b'l') if bytes.get(at + 2) == Some(&b'l') => 3,
            _ => continue,
        };
        if at + 1 < 64 {
            cuts &= !(((1_u64 << (length - 1)) - 1) << (at + 1));
        }
        if at + length < limit {
            cuts |= 1_u64 << (at + length);
        } else if at + length == 64 && limit == 64 {
            end64 = true;
        }
    }
    (cuts >> 1) | (u64::from(end64) << 63)
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

impl<'t, I> Pieces<'_, 't, I>
where
    I: Iterator<Item = Result<&'t str, fancy_regex::Error>>,
{
    #[inline(always)]
    pub(crate) fn next_rank<'a>(
        &mut self,
        ranks: &'a EncoderRanks,
    ) -> Option<Result<(&'t str, Option<&'a Rank>), fancy_regex::Error>> {
        match self {
            Self::Original(original) => original
                .next()
                .map(|result| result.map(|piece| (piece, ranks.get(piece.as_bytes())))),
            Self::Regular(regular) => regular.next_rank(ranks).map(Ok),
        }
    }
}

impl<'t, I> Iterator for Pieces<'_, 't, I>
where
    I: Iterator<Item = Result<&'t str, fancy_regex::Error>>,
{
    type Item = Result<&'t str, fancy_regex::Error>;

    #[inline(always)]
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
    ascii_ends: u64,
    ascii_start: usize,
    vector_ascii: bool,
}

impl<'r, 't> RegularPieces<'r, 't> {
    pub(crate) fn new(regex: &'r RegularRegex, text: &'t str) -> Self {
        Self {
            regex,
            text,
            input: Input::new(text).anchored(Anchored::Yes),
            cache: regex.caches.get(),
            ascii_ends: 0,
            ascii_start: 0,
            vector_ascii: {
                #[cfg(target_arch = "x86_64")]
                {
                    regex.ascii_guarded && std::is_x86_feature_detected!("avx2")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            },
        }
    }
}

impl<'t> Iterator for RegularPieces<'_, 't> {
    type Item = &'t str;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.input.start();
        if start == self.text.len() {
            return None;
        }
        if self.ascii_ends == 0 && self.vector_ascii && self.text.len() - start >= 64 {
            #[cfg(target_arch = "x86_64")]
            {
                // The feature and the two 32-byte load bounds were checked above.
                self.ascii_ends = unsafe { ascii_block_ends(&self.text[start..]) };
                self.ascii_start = start;
            }
        }
        if self.ascii_ends != 0 {
            let end = self.ascii_start + self.ascii_ends.trailing_zeros() as usize + 1;
            self.ascii_ends &= self.ascii_ends - 1;
            self.input.set_start(end);
            return Some(&self.text[start..end]);
        }
        if self.regex.ascii_guarded
            && let Some(length) = ascii_piece_len(&self.text[start..])
        {
            let end = start + length;
            self.input.set_start(end);
            return Some(&self.text[start..end]);
        }
        self.next_unicode(start)
    }
}

impl<'t> RegularPieces<'_, 't> {
    #[inline(always)]
    fn next_rank<'a>(&mut self, ranks: &'a EncoderRanks) -> Option<(&'t str, Option<&'a Rank>)> {
        let start = self.input.start();
        let piece = self.next()?;
        let length = piece.len();
        let rank = if (1..=7).contains(&length) && self.text.len() - start >= 8 {
            // All eight bytes are in this ordinary segment. The key masks off
            // every byte beyond the already established piece boundary.
            let word = u64::from_le_bytes(self.text.as_bytes()[start..start + 8].try_into().unwrap());
            ranks.get_short_word(word, length)
        } else {
            ranks.get(piece.as_bytes())
        };
        Some((piece, rank))
    }

    // Keep the complete DFA/error/whitespace path outside the common ASCII loop.
    #[inline(never)]
    fn next_unicode(&mut self, start: usize) -> Option<&'t str> {
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

    use super::{
        GUARDED_PATTERN, RegularPieces, RegularRegex, build_regular_regex, compile_if_guarded,
        is_guarded_whitespace,
    };
    use crate::{CoreBPE, HashMap, Rank};

    const GUARDED_WHITESPACE: &str = r"\x{9}-\x{d}\x{20}\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}-\x{2029}\x{202f}\x{205f}\x{3000}";

    #[test]
    fn boundary_bitsets_match_all_ascii_pairs_across_blocks() {
        let (original, regular) = matchers();
        for left in 0..128_u8 {
            for right in 0..128_u8 {
                let pair = String::from_utf8(vec![left, right]).unwrap();
                let text = format!("{}{}é 中", "x ".repeat(16), pair.repeat(40));
                assert_spans_equal(&original, &regular, &text);
            }
        }
    }

    #[test]
    fn boundary_bitsets_preserve_contractions_and_whitespace_at_block_edges() {
        let (original, regular) = matchers();
        for padding in 0..=65 {
            for suffix in [
                "'s",
                "'t",
                "'m",
                "'d",
                "'re",
                "'ve",
                "'ll",
                "''s",
                " 's",
                "  word",
                " \tword",
                "\t word",
                "\n\t word",
                "'shopping",
                "'s's",
                "é",
                " 中",
            ] {
                let text = format!("{}{}", "x ".repeat(padding), suffix);
                assert_spans_equal(&original, &regular, &text);
                assert_spans_equal(&original, &regular, &format!("{text} suffix é"));
            }
        }
    }

    #[test]
    fn boundary_bitsets_match_seeded_mixed_ascii() {
        let (original, regular) = matchers();
        let alphabet = b"aB09 '\t\n!?rmldesv\0\x7f";
        let mut seed = 51702_u64;
        for length in 64..576 {
            let mut text = String::with_capacity(length + 8);
            for _ in 0..length {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                text.push(alphabet[(seed >> 32) as usize % alphabet.len()] as char);
            }
            assert_spans_equal(&original, &regular, &text);
            text.insert_str(length / 2, "é 中");
            assert_spans_equal(&original, &regular, &text);
        }
    }

    #[test]
    fn bulk_ascii_boundaries_preserve_all_bytes_lanes_and_unicode_edges() {
        let (original, regular) = matchers();
        for byte in 0..128_u8 {
            for offset in [0, 1, 30, 31, 32, 33, 62, 63, 64] {
                let repeated = char::from(byte).to_string().repeat(90);
                for tail in [" 's  x", "\té中\u{3000}a", "1١ !\u{301}"] {
                    let text = format!("{}{}{}", "a ".repeat(offset), repeated, tail);
                    assert_spans_equal(&original, &regular, &text);
                    let mut scalar = RegularPieces::new(&regular, &text);
                    scalar.vector_ascii = false;
                    assert_eq!(
                        RegularPieces::new(&regular, &text).collect::<Vec<_>>(),
                        scalar.collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn bulk_ascii_contractions_and_whitespace_cross_every_block_position() {
        let (original, regular) = matchers();
        for prefix in 0..130 {
            for body in [
                "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'rr", " 's", "!'s", "  \tX", "   X",
                " \té", "\r\n",
            ] {
                let text = format!("{}{}{}", "x ".repeat(prefix), body, " a".repeat(100));
                assert_spans_equal(&original, &regular, &text);
            }
        }
    }

    #[test]
    fn ascii_pairs_contractions_and_unicode_boundaries_match_fancy_regex() {
        let (original, regular) = matchers();
        assert!(regular.ascii_guarded && regular.clone().ascii_guarded);
        assert!(!build_regular_regex("a").unwrap().ascii_guarded);
        for left in 0..128_u8 {
            for right in 0..128_u8 {
                let prefix = String::from_utf8(vec![left, right]).unwrap();
                for tail in [" 're\téX", "123!\u{3000}x"] {
                    assert_spans_equal(&original, &regular, &format!("{prefix}{tail}"));
                }
            }
        }
        for text in ["é", "aé", "1١", "!\u{301}", " \u{3000}"] {
            assert!(super::ascii_piece_len(text).is_none());
        }
    }

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
        let mut ranks: HashMap<Vec<u8>, Rank> = (0_u16..256)
            .map(|byte| (vec![byte as u8], byte as Rank))
            .collect();
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
        let class = whitespace
            .strip_prefix('[')
            .unwrap()
            .strip_suffix("]+")
            .unwrap();
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
            '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
            '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
            '\u{205f}', '\u{3000}', '\u{1c}', '\u{1f}', '\u{200b}', '\u{feff}', 'é', '३',
            '\u{301}', '😀', '\'', '!',
        ] {
            for suffix in ["", "a", "1", "!", "'s", "'re", "'S", "<S>"] {
                let text = format!("a{scalar} {scalar}{suffix}");
                for piece in RegularPieces::new(&regular, &text) {
                    let last = piece.chars().next_back().unwrap();
                    assert_eq!(
                        is_guarded_whitespace(last),
                        piece.chars().all(is_guarded_whitespace),
                        "piece {piece:?}"
                    );
                }
                assert_spans_equal(&original, &regular, &text);
            }
        }
    }

    #[test]
    fn contractions_restart_at_withheld_scalar() {
        let (original, regular) = matchers();
        for text in [
            "  's",
            "\t 's",
            " \t's",
            "\u{a0} 's",
            "  're",
            "  'll",
            "  'S",
            "a  'd!",
            "  't",
            "  've",
            "  'm",
            "   '!",
            " 's",
            "\t's",
        ] {
            assert_spans_equal(&original, &regular, text);
        }
        assert_eq!(
            RegularPieces::new(&regular, "  's").collect::<Vec<_>>(),
            [" ", " '", "s"]
        );
        assert_eq!(
            RegularPieces::new(&regular, "\t 's").collect::<Vec<_>>(),
            ["\t", " '", "s"]
        );
    }

    #[test]
    fn eof_and_single_scalar_whitespace_make_progress() {
        let (original, regular) = matchers();
        for text in [
            "",
            " ",
            "  ",
            "\t",
            "\n\t ",
            "\u{a0}",
            "\u{2003}\u{3000}",
            "a \t\u{a0}",
            "\t!",
            "\u{2003}a",
            " \u{2003}a",
            "\u{2003} a",
            "\r\n!",
        ] {
            assert_spans_equal(&original, &regular, text);
        }
        assert_eq!(
            RegularPieces::new(&regular, " \t\u{a0}").collect::<Vec<_>>(),
            [" \t\u{a0}"]
        );
    }

    #[test]
    fn explicit_unicode_classes_and_utf8_boundaries_match() {
        let (original, regular) = matchers();
        for text in [
            "élève 中文 １２३",
            "a\u{301} 𝟠!😀",
            "\u{85}\u{a0}'s",
            "\u{1680}\u{2003}x",
            "\u{2028}\u{2029}'re",
            "\u{202f}\u{205f}7",
            "\u{3000} 😀",
            "\u{200b}\u{feff}x",
            "\u{0}a\u{10ffff}",
            "\u{1c}\u{1f} a",
            "\u{e000}\u{fdd0}",
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
        assert_eq!(
            RegularPieces::new(&regular, "x \t\u{a0}").collect::<Vec<_>>(),
            ["x", " \t\u{a0}"]
        );
    }

    #[test]
    fn bounded_exhaustive_actual_pattern_spans_match() {
        fn visit(
            original: &fancy_regex::Regex,
            regular: &RegularRegex,
            text: &mut String,
            remaining: usize,
        ) -> usize {
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
        let allowed_sets = [
            HashSet::new(),
            HashSet::from(["<S>"]),
            HashSet::from(["<S>", "<T>"]),
        ];
        for text in [
            "",
            "  's",
            "\t 's",
            "a  ",
            "é \u{3000}'re😀",
            "  <S>'s",
            "\t <S>  's<T> ",
            "<S><T>",
            "<S> \u{2003}<T>",
            "  <S>  ",
            "x<T> <S>'s<T>",
        ] {
            assert_eq!(
                candidate.encode_ordinary(text),
                original.encode_ordinary(text),
                "{text:?}"
            );
            for allowed in &allowed_sets {
                assert_eq!(
                    candidate.encode(text, allowed).unwrap(),
                    original.encode(text, allowed).unwrap(),
                    "{text:?}, {allowed:?}"
                );
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
        assert_eq!(
            RegularPieces::new(&simple, "a a").collect::<Vec<_>>(),
            ["a"]
        );
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
        let texts = [
            "",
            "aa11!!bb  's\u{3000}\u{a0}é  ",
            "\t 's",
            "é中文😀\0",
            " a  ",
        ];
        let original = fancy_regex::Regex::new(GUARDED_PATTERN).unwrap();
        let expected: Vec<Vec<String>> = texts
            .iter()
            .map(|text| {
                original
                    .find_iter(text)
                    .map(|m| m.unwrap().as_str().to_owned())
                    .collect()
            })
            .collect();
        let clone = regular.as_ref().clone();
        assert!(Arc::ptr_eq(&regular.dfa, &clone.dfa));
        // Keep one pooled cache checked out while another iterator comes and goes.
        let mut held = RegularPieces::new(&regular, texts[1]);
        assert_eq!(held.next(), Some("aa"));
        let mut early = RegularPieces::new(&regular, texts[2]);
        assert_eq!(early.next(), Some("\t"));
        drop(early);
        for (text, wanted) in texts.iter().zip(&expected) {
            assert_eq!(
                RegularPieces::new(&clone, text).collect::<Vec<_>>(),
                *wanted
            );
        }
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let regular = Arc::clone(&regular);
                let expected = expected.clone();
                std::thread::spawn(move || {
                    for _ in 0..64 {
                        for (text, wanted) in texts.iter().zip(&expected) {
                            assert_eq!(
                                RegularPieces::new(&regular, text).collect::<Vec<_>>(),
                                *wanted
                            );
                        }
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
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

#[cfg(test)]
mod bounded_key_tests {
    use super::*;
    use rustc_hash::FxHashMap as HashMap;

    #[test]
    fn ranked_boundaries_match_scalar_rank_lookup_for_all_offsets_and_tails() {
        let ranks = EncoderRanks::from(HashMap::from_iter([
            (b"a".to_vec(), 0), (b" a".to_vec(), Rank::MAX),
            (b"abcdefg".to_vec(), 19), (b"'re".to_vec(), 31),
            ("é".as_bytes().to_vec(), 37), (b" ".to_vec(), 39),
            (b"42".to_vec(), 42), (b"!?".to_vec(), 44),
        ]));
        let regex = compile_if_guarded(GUARDED_PATTERN).unwrap().unwrap();
        for prefix in 0..66 {
            for tail in 0..10 {
                for sample in ["", "a! a! abcdefg!", "aé! 中文 test 🙂", "'re aaa!!", "  a\t\nend", "abcdefghi!", " aaaé", "42!?", "\0a\x7f!"] {
                    let text = format!("{}{}{}", "x ".repeat(prefix), sample, " ".repeat(tail));
                    let expected: Vec<_> = RegularPieces::new(&regex, &text).collect();
                    let mut ranked = RegularPieces::new(&regex, &text);
                    let mut actual = Vec::new();
                    while let Some((piece, rank)) = ranked.next_rank(&ranks) {
                        assert_eq!(rank, ranks.get(piece.as_bytes()), "{text:?}: {piece:?}");
                        actual.push(piece);
                    }
                    assert_eq!(actual, expected);
                    assert!(ranked.next_rank(&ranks).is_none());
                }
            }
        }
    }

    #[test]
    fn arbitrary_pattern_ranked_iterator_preserves_original_results() {
        let ranks = EncoderRanks::from(HashMap::from_iter([(b"a".to_vec(), Rank::MAX)]));
        let regex = fancy_regex::Regex::new(r"a|[^a]+").unwrap();
        let text = "aé42 ! a";
        let expected: Vec<_> = regex.find_iter(text).map(|x| x.unwrap().as_str()).collect();
        let mut pieces = Pieces::Original(regex.find_iter(text).map(|x| x.map(|m| m.as_str())));
        let mut actual = Vec::new();
        while let Some(result) = pieces.next_rank(&ranks) {
            let (piece, rank) = result.unwrap();
            assert_eq!(rank, ranks.get(piece.as_bytes()));
            actual.push(piece);
        }
        assert_eq!(actual, expected);
    }
}
