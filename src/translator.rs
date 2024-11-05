use std::collections::BTreeMap;

#[derive(Debug)]
pub enum TranslationError {
    Error
}

pub trait U32ToChar {
    fn u32_to_char(&self, c: u32) -> Result<char, TranslationError>;

    fn u32_array_to_chars(&self, cs: &[u32]) -> Result<Vec<char>, TranslationError>;
}

pub trait CharToU32 {
    fn char_to_u32(&self, c: char) -> Result<u32, TranslationError>;

    fn char_array_to_u32s(&self, cs: &[char]) -> Result<Vec<u32>, TranslationError>;
}

pub struct Translator {
    forward_map: BTreeMap<u32, char>,
    reverse_map: BTreeMap<char, u32>
}

impl Translator {
    pub fn new(unique_chars: &[&u8]) -> Result<Translator, TranslationError> {
        Ok(Self {
            forward_map: (0..).zip(unique_chars.iter().map(|c| **c as char)).collect(),
            reverse_map: (0..).zip(unique_chars.iter().map(|c| **c as char)).map(|(i, c)| (c, i)).collect(),
        })
    }
}

impl U32ToChar for Translator {
    fn u32_to_char(&self, c: u32) -> Result<char, TranslationError> {
        self.forward_map.get(&c).map(|v| *v).ok_or(TranslationError::Error)
    }

    fn u32_array_to_chars(&self, cs: &[u32]) -> Result<Vec<char>, TranslationError> {
        cs.iter().map(|c| self.u32_to_char(*c)).collect()
    }
}

impl CharToU32 for Translator {
    fn char_to_u32(&self, c: char) -> Result<u32, TranslationError> {
        self.reverse_map.get(&c).map(|v| *v).ok_or(TranslationError::Error)
    }

    fn char_array_to_u32s(&self, cs: &[char]) -> Result<Vec<u32>, TranslationError> {
        cs.iter().map(|c| self.char_to_u32(*c)).collect()
    }
}