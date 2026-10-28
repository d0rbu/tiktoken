//! Whole-record source batches. Unsupported JSON is replayed before encoding.
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use serde_json::value::RawValue;

pub(crate) struct Batch {
    pub texts: Vec<String>,
    pub provenance: Vec<Option<String>>,
}

pub(crate) enum ReadError {
    Io(io::Error),
    Declined,
}

impl From<io::Error> for ReadError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub(crate) struct Cursor {
    input: Option<BufReader<File>>,
    position: u64,
    stop: Option<u64>,
    split: String,
    replay: VecDeque<Vec<u8>>,
    pending_error: Option<io::Error>,
    finished: bool,
}

impl Cursor {
    pub fn open(path: &Path, split: String, start: u64, stop: Option<u64>) -> io::Result<Self> {
        let mut input = BufReader::new(File::open(path)?);
        let mut position = start;
        if start > 0 {
            input.seek(SeekFrom::Start(start - 1))?;
            position = start - 1 + input.skip_until(b'\n')? as u64;
        }
        Ok(Self {
            input: Some(input),
            position,
            stop,
            split,
            replay: VecDeque::new(),
            pending_error: None,
            finished: false,
        })
    }

    pub fn close(&mut self) {
        self.input.take();
        self.replay.clear();
        self.pending_error.take();
        self.finished = true;
    }

    pub fn raw_line(&mut self) -> io::Result<Option<Vec<u8>>> {
        if let Some(line) = self.replay.pop_front() {
            return Ok(Some(line));
        }
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        if self.finished || self.stop.is_some_and(|stop| self.position >= stop) {
            self.input.take();
            self.finished = true;
            return Ok(None);
        }
        let mut line = Vec::new();
        let Some(input) = self.input.as_mut() else {
            return Ok(None);
        };
        match input.read_until(b'\n', &mut line) {
            Ok(0) => {
                self.finished = true;
                self.input.take();
                Ok(None)
            }
            Ok(size) => {
                self.position += size as u64;
                Ok(Some(line))
            }
            Err(error) => {
                self.finished = true;
                self.input.take();
                Err(error)
            }
        }
    }

    pub fn batch(&mut self, provenance: bool) -> Result<Option<Batch>, ReadError> {
        let mut result = Batch {
            texts: Vec::new(),
            provenance: Vec::new(),
        };
        let mut raw = VecDeque::new();
        let mut bytes = 0;
        loop {
            let line = match self.raw_line() {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(error) if result.texts.is_empty() => return Err(ReadError::Io(error)),
                Err(error) => {
                    // The existing producer emits its selected partial batch first.
                    self.pending_error = Some(error);
                    break;
                }
            };
            // Borrow exact JSON operands rather than normalizing source metadata.
            let parsed = serde_json::from_slice::<HashMap<String, Box<RawValue>>>(&line);
            let document = parsed.ok().and_then(|row| {
                let split: String = serde_json::from_str(row.get("split")?.get()).ok()?;
                if split != self.split {
                    return Some(None);
                }
                let text: String = serde_json::from_str(row.get("text")?.get()).ok()?;
                let original = if provenance {
                    row.get("token_provenance").map(|v| v.get().to_owned())
                } else {
                    None
                };
                Some(Some((text, original)))
            });
            match document {
                None => {
                    // No encoding has happened for this batch. Python consumes
                    // these original bytes and the remaining cursor exactly once.
                    raw.push_back(line);
                    raw.append(&mut self.replay);
                    self.replay = raw;
                    return Err(ReadError::Declined);
                }
                // A fully parsed excluded row has no downstream effect. It
                // must not accumulate in the replay buffer while scanning a
                // long excluded prefix between selected documents.
                Some(None) => (),
                Some(Some((text, original))) => {
                    if !result.texts.is_empty() && bytes + text.len() > 1024 * 1024 {
                        self.replay.push_front(line);
                        break;
                    }
                    bytes += text.len();
                    result.texts.push(text);
                    if provenance {
                        result.provenance.push(original);
                    }
                    raw.push_back(line);
                    if result.texts.len() == 64 || bytes >= 1024 * 1024 {
                        break;
                    }
                }
            }
        }
        if result.texts.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new(bytes: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "rmcp-stream-source-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            std::io::Write::write_all(&mut file, bytes).unwrap();
            Self(path)
        }
        fn cursor(&self) -> Cursor {
            Cursor::open(&self.0, "train".into(), 0, None).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).unwrap();
        }
    }

    fn batch(cursor: &mut Cursor, provenance: bool) -> Option<Batch> {
        match cursor.batch(provenance) {
            Ok(result) => result,
            Err(_) => panic!("unexpected source decline/error"),
        }
    }

    #[test]
    fn ordinary_rows_filter_and_preserve_exact_provenance_operands() {
        let fixture = Fixture::new(r#"{"split":"eval","text":null}
{"split":"train","text":"é😀","token_provenance":{"original_source_tokens":9007199254740993,"unknown":-0.0}}
{"split":"eval"}
{"split":"train","text":"","token_provenance":null}"#.as_bytes());
        let mut cursor = fixture.cursor();
        let out = batch(&mut cursor, true).unwrap();
        assert_eq!(out.texts, ["é😀", ""]);
        assert_eq!(
            out.provenance,
            [
                Some("{\"original_source_tokens\":9007199254740993,\"unknown\":-0.0}".into()),
                Some("null".into())
            ]
        );
        assert!(batch(&mut cursor, true).is_none());
    }

    #[test]
    fn every_source_byte_cut_selects_each_whole_record_once() {
        let data = "{\"split\":\"train\",\"text\":\"one 😀\"}\r\n{\"split\":\"eval\"}\n{\"split\":\"train\",\"text\":\"two\"}";
        let fixture = Fixture::new(data.as_bytes());
        for width in [1, 7, 64, 1024] {
            let mut texts = Vec::new();
            for start in (0..data.len()).step_by(width) {
                let mut cursor = Cursor::open(
                    &fixture.0,
                    "train".into(),
                    start as u64,
                    Some((start + width).min(data.len()) as u64),
                )
                .unwrap();
                while let Some(out) = batch(&mut cursor, false) {
                    assert!(out.provenance.is_empty());
                    texts.extend(out.texts);
                }
            }
            assert_eq!(texts, ["one 😀", "two"]);
        }
    }

    #[test]
    fn unsupported_json_replays_current_batch_without_reencoding_prior_batches() {
        for bad in [
            b"{broken}".as_slice(),
            br#"{"split":"train","text":"\ud800"}"#,
            b"[]",
            br#"{"text":"missing split"}"#,
            br#"{"split":"train","text":null}"#,
        ] {
            let good = b"{\"split\":\"train\",\"text\":\"ok\"}\n";
            let mut data = good.repeat(65);
            data.extend(bad);
            data.push(b'\n');
            let fixture = Fixture::new(&data);
            let mut cursor = fixture.cursor();
            assert_eq!(batch(&mut cursor, false).unwrap().texts.len(), 64);
            assert!(matches!(cursor.batch(false), Err(ReadError::Declined)));
            let mut replay = Vec::new();
            while let Some(line) = cursor.raw_line().unwrap() {
                replay.extend(line)
            }
            let mut expected = good.to_vec();
            expected.extend(bad);
            expected.push(b'\n');
            assert_eq!(replay, expected);
        }
    }

    #[test]
    fn excluded_rows_do_not_accumulate_in_fallback_replay() {
        let excluded = b"{\"split\":\"eval\",\"text\":null}\n";
        let selected = b"{\"split\":\"train\",\"text\":\"ok\"}\n";
        let bad = b"{\"split\":\"train\",\"text\":null}\n";
        let mut data = excluded.repeat(4096);
        data.extend(selected);
        data.extend(excluded.repeat(4096));
        data.extend(bad);
        let fixture = Fixture::new(&data);
        let mut cursor = fixture.cursor();
        assert!(matches!(cursor.batch(false), Err(ReadError::Declined)));
        assert_eq!(cursor.replay.len(), 2);
        assert_eq!(cursor.raw_line().unwrap().unwrap(), selected);
        assert_eq!(cursor.raw_line().unwrap().unwrap(), bad);
        assert!(cursor.raw_line().unwrap().is_none());
    }

    #[test]
    fn aggregation_limits_keep_oversized_singletons_whole() {
        let texts = [
            "a".repeat(17),
            "b".repeat(1024 * 1024),
            "c".repeat(2),
            "".into(),
        ];
        let data: Vec<u8> = texts
            .iter()
            .flat_map(|text| format!("{{\"split\":\"train\",\"text\":\"{text}\"}}\n").into_bytes())
            .collect();
        let fixture = Fixture::new(&data);
        let mut cursor = fixture.cursor();
        assert_eq!(batch(&mut cursor, false).unwrap().texts, [texts[0].clone()]);
        assert_eq!(batch(&mut cursor, false).unwrap().texts, [texts[1].clone()]);
        assert_eq!(batch(&mut cursor, false).unwrap().texts, texts[2..]);
        assert!(batch(&mut cursor, false).is_none());
    }

    #[test]
    fn read_error_follows_complete_selected_prefix_and_close_releases_replay() {
        let fixture = Fixture::new(b"");
        let mut cursor = fixture.cursor();
        cursor
            .replay
            .push_back(b"{\"split\":\"train\",\"text\":\"ok\"}\n".to_vec());
        cursor.pending_error = Some(io::Error::from_raw_os_error(5));
        assert_eq!(batch(&mut cursor, false).unwrap().texts, ["ok"]);
        match cursor.batch(false) {
            Err(ReadError::Io(error)) => assert_eq!(error.raw_os_error(), Some(5)),
            _ => panic!("deferred I/O error missing"),
        }
        cursor.replay.push_back(b"unused".to_vec());
        cursor.pending_error = Some(io::Error::from_raw_os_error(5));
        cursor.close();
        assert!(cursor.input.is_none());
        assert!(cursor.raw_line().unwrap().is_none());
        let mut stopped = Cursor::open(&fixture.0, "train".into(), 0, Some(0)).unwrap();
        assert!(batch(&mut stopped, false).is_none());
        assert!(stopped.input.is_none());
    }
}
