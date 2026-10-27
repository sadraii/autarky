//! Checked, bounded GGUF metadata parsing. Tensor data is never read.

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Clone, Serialize)]
pub struct GgufInfo {
    pub model_name: Option<String>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
    /// Preserve unsupported codes without claiming a known quantization.
    pub file_type: Option<u32>,
    pub is_moe: bool,
    pub expert_count: Option<u32>,
    pub block_count: Option<u32>,
    pub parameters_billions: Option<f64>,
    pub context_length: Option<u32>,
}

// Bound both allocation and work, independently of the untrusted file size.
const MAX_METADATA_BYTES: u64 = 256 * 1024 * 1024;
const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_KEY_BYTES: u64 = 65_535;
const MAX_ARRAY_DEPTH: usize = 16;

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

struct MetadataReader<R> {
    input: R,
    length: u64,
    position: u64,
    endian: Endian,
    values_left: u64,
}

impl<R: Read + Seek> MetadataReader<R> {
    fn end(&self, bytes: u64) -> Result<u64> {
        let end = self
            .position
            .checked_add(bytes)
            .context("GGUF offset overflow")?;
        ensure!(
            end <= self.length,
            "Truncated GGUF metadata at byte {}",
            self.position
        );
        ensure!(
            end <= MAX_METADATA_BYTES,
            "GGUF metadata exceeds byte limit"
        );
        Ok(end)
    }

    fn read<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.end(N as u64)?;
        let mut bytes = [0; N];
        self.input
            .read_exact(&mut bytes)
            .context("Truncated GGUF metadata")?;
        self.position = end;
        Ok(bytes)
    }

    fn skip(&mut self, bytes: u64) -> Result<()> {
        let end = self.end(bytes)?;
        self.input.seek(SeekFrom::Start(end))?;
        self.position = end;
        Ok(())
    }

    fn u16(&mut self) -> Result<u16> {
        let bytes = self.read()?;
        Ok(match self.endian {
            Endian::Little => u16::from_le_bytes(bytes),
            Endian::Big => u16::from_be_bytes(bytes),
        })
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.read()?;
        Ok(match self.endian {
            Endian::Little => u32::from_le_bytes(bytes),
            Endian::Big => u32::from_be_bytes(bytes),
        })
    }

    fn u64(&mut self) -> Result<u64> {
        let bytes = self.read()?;
        Ok(match self.endian {
            Endian::Little => u64::from_le_bytes(bytes),
            Endian::Big => u64::from_be_bytes(bytes),
        })
    }

    fn string(&mut self, limit: u64) -> Result<String> {
        let length = self.u64()?;
        ensure!(length <= limit, "GGUF string exceeds length limit");
        let end = self.end(length)?;
        let length = usize::try_from(length).context("GGUF string length does not fit memory")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .context("Cannot allocate GGUF string")?;
        bytes.resize(length, 0);
        self.input
            .read_exact(&mut bytes)
            .context("Truncated GGUF string")?;
        self.position = end;
        String::from_utf8(bytes).context("GGUF string is not UTF-8")
    }

    fn value(&mut self, kind: u32, keep_string: bool, depth: usize) -> Result<Value> {
        self.values_left = self
            .values_left
            .checked_sub(1)
            .context("GGUF value count exceeds limit")?;
        Ok(match kind {
            0 => Value::Unsigned(u64::from(self.read::<1>()?[0])),
            1 => Value::Signed(i64::from(i8::from_ne_bytes(self.read()?))),
            2 => Value::Unsigned(u64::from(self.u16()?)),
            3 => Value::Signed(i64::from(self.u16()? as i16)),
            4 => Value::Unsigned(u64::from(self.u32()?)),
            5 => Value::Signed(i64::from(self.u32()? as i32)),
            6 => Value::Float(f64::from(f32::from_bits(self.u32()?))),
            7 => {
                ensure!(self.read::<1>()?[0] <= 1, "Invalid GGUF boolean");
                Value::Other
            }
            8 if keep_string => Value::String(self.string(MAX_STRING_BYTES)?),
            8 => {
                let length = self.u64()?;
                self.skip(length)?;
                Value::Other
            }
            9 => {
                ensure!(depth < MAX_ARRAY_DEPTH, "GGUF array nesting exceeds limit");
                let element_type = self.u32()?;
                let count = self.u64()?;
                ensure!(element_type <= 12, "Unknown GGUF array type {element_type}");
                if let Some(width) = value_width(element_type) {
                    let bytes = count
                        .checked_mul(width)
                        .context("GGUF array size overflow")?;
                    self.skip(bytes)?;
                } else {
                    // Variable-width elements must also have bounded iteration counts.
                    ensure!(count <= self.values_left, "GGUF array count exceeds limit");
                    for _ in 0..count {
                        self.value(element_type, false, depth + 1)?;
                    }
                }
                Value::Other
            }
            10 => Value::Unsigned(self.u64()?),
            11 => Value::Signed(self.u64()? as i64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            _ => bail!("Unknown GGUF value type {kind}"),
        })
    }
}

fn value_width(kind: u32) -> Option<u64> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

enum Value {
    String(String),
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Other,
}

impl Value {
    fn unsigned(self, key: &str) -> Result<u64> {
        match self {
            Self::Unsigned(value) => Ok(value),
            Self::Signed(value) => {
                u64::try_from(value).with_context(|| format!("Negative GGUF {key}"))
            }
            _ => bail!("GGUF {key} must be an integer"),
        }
    }

    fn count(self, key: &str) -> Result<u32> {
        u32::try_from(self.unsigned(key)?).with_context(|| format!("GGUF {key} exceeds u32"))
    }
}

/// Parse version 2/3 metadata and stop before the tensor table and tensor bytes.
pub fn parse_gguf(path: &str) -> Result<GgufInfo> {
    let file = std::fs::File::open(path).with_context(|| format!("Cannot open GGUF {path}"))?;
    let length = file.metadata()?.len();
    parse_metadata(file, length)
}

fn parse_metadata<R: Read + Seek>(input: R, length: u64) -> Result<GgufInfo> {
    let mut reader = MetadataReader {
        input,
        length,
        position: 0,
        endian: Endian::Little,
        values_left: MAX_METADATA_ENTRIES,
    };
    let magic = reader.read::<4>()?;
    ensure!(magic == *b"GGUF" || magic == *b"FUGG", "Not a GGUF file");
    let version = reader.read::<4>()?;
    reader.endian = if magic == *b"GGUF" && matches!(u32::from_le_bytes(version), 2 | 3) {
        Endian::Little
    } else if u32::from_be_bytes(version) == 3 {
        Endian::Big
    } else {
        bail!("Unsupported GGUF version {}", u32::from_le_bytes(version));
    };
    let _tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    ensure!(
        metadata_count <= MAX_METADATA_ENTRIES,
        "GGUF metadata count exceeds limit"
    );
    let mut info = GgufInfo {
        model_name: None,
        architecture: None,
        quantization: None,
        file_type: None,
        is_moe: false,
        expert_count: None,
        block_count: None,
        parameters_billions: None,
        context_length: None,
    };
    for _ in 0..metadata_count {
        let key = reader.string(MAX_KEY_BYTES)?;
        ensure!(!key.is_empty(), "Empty GGUF metadata key");
        let kind = reader.u32()?;
        let keep_string = matches!(key.as_str(), "general.name" | "general.architecture");
        let value = reader
            .value(kind, keep_string, 0)
            .with_context(|| format!("Invalid GGUF {key}"))?;
        match key.as_str() {
            "general.name" => match value {
                Value::String(name) => info.model_name = Some(name),
                _ => bail!("GGUF general.name must be a string"),
            },
            "general.architecture" => match value {
                Value::String(architecture) => info.architecture = Some(architecture),
                _ => bail!("GGUF general.architecture must be a string"),
            },
            "general.file_type" => {
                let code = value.count(&key)?;
                info.file_type = Some(code);
                info.quantization = file_type_to_quant(code).map(str::to_string);
            }
            "general.parameter_count" => {
                let count = match value {
                    Value::Float(count) => {
                        ensure!(
                            count.is_finite() && count >= 0.0,
                            "Invalid GGUF parameter count"
                        );
                        count
                    }
                    value => value.unsigned(&key)? as f64,
                };
                info.parameters_billions = Some(count / 1_000_000_000.0);
            }
            _ if key.ends_with(".expert_count") => info.expert_count = Some(value.count(&key)?),
            _ if key.ends_with(".block_count") => info.block_count = Some(value.count(&key)?),
            _ if key.ends_with(".context_length") => info.context_length = Some(value.count(&key)?),
            _ => {}
        }
    }
    info.is_moe = info.expert_count.is_some_and(|count| count > 0);
    Ok(info)
}

/// llama.cpp's llama_ftype (not ggml_type). Removed enum slots remain explicit.
fn file_type_to_quant(ftype: u32) -> Option<&'static str> {
    Some(match ftype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        39 => "NVFP4",
        40 => "Q1_0",
        // LLAMA_FTYPE_GUESSED is a sentinel, not a quantization identity.
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // The logical file may be huge, but every permitted byte is held here.
    // Any tensor read, buffering, or oversized seek fails immediately.
    struct MetadataOnly(Cursor<Vec<u8>>);

    impl Read for MetadataOnly {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            assert!(
                self.0.position() + buffer.len() as u64 <= self.0.get_ref().len() as u64,
                "read beyond metadata"
            );
            self.0.read(buffer)
        }
    }

    impl Seek for MetadataOnly {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            match position {
                SeekFrom::Start(end) => assert!(end <= self.0.get_ref().len() as u64),
                _ => panic!("metadata parser must use checked absolute seeks"),
            }
            self.0.seek(position)
        }
    }

    fn header(metadata_count: u64) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(metadata_count.to_le_bytes());
        bytes
    }

    #[test]
    fn tensor_bytes_are_never_read_or_buffered() {
        let mut bytes = header(1);
        let key = b"general.name";
        let name = b"tiny_model";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key);
        bytes.extend(8u32.to_le_bytes());
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name);
        let input = MetadataOnly(Cursor::new(bytes));
        let info = parse_metadata(input, 1 << 40).unwrap();
        assert_eq!(info.model_name.as_deref(), Some("tiny_model"));
        assert_eq!(info.quantization, None);
    }

    #[test]
    fn metadata_byte_budget_rejects_arrays_and_strings_before_seeking() {
        for kind in [8u32, 9] {
            let mut bytes = header(1);
            let key = b"unused";
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key);
            bytes.extend(kind.to_le_bytes());
            if kind == 9 {
                bytes.extend(0u32.to_le_bytes());
            }
            bytes.extend((MAX_METADATA_BYTES + 1).to_le_bytes());
            assert!(bytes.len() < 64);
            let input = MetadataOnly(Cursor::new(bytes));
            let error = parse_metadata(input, 1 << 40).unwrap_err();
            assert!(format!("{error:#}").contains("metadata exceeds byte limit"));
        }
    }
}
