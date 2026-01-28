use crate::ply_splat_core;
use crate::splats::Splats;
use crate::splats_parser::{ParseError, SplatsParser};

pub struct PlyParser;

impl SplatsParser for PlyParser {
    fn parse(bytes: &[u8]) -> Result<Splats, ParseError> {
        ply_splat_core::parse_splat_ply_core(bytes).map_err(|e| ParseError::msg(e.to_string()))
    }
}

