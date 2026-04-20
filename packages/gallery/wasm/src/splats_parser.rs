use crate::splats::Splats;

#[derive(Debug)]
pub enum ParseError {
    Msg(String),
}

impl ParseError {
    pub fn msg<S: Into<String>>(s: S) -> Self {
        ParseError::Msg(s.into())
    }
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::Msg(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for ParseError {}

pub trait SplatsParser {
    fn parse(bytes: &[u8]) -> Result<Splats, ParseError>;
}

