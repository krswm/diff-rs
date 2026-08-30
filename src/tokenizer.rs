// GPT-2 Inference with tenferro
// Copyright (C) 2026  Kurosawa Mutsumi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use std::collections::HashMap;
use std::error::Error;

use crate::model::Model;

// CLIP has a unique encoding.
// e.g.: 'à' (U+00E0) is encoded as "Ãł" (U+00C3 0+0142)
fn encode_unique_encoding(text: &str) -> String {
    text.bytes()
        .map(|x| {
            let y = x as u32;
            TryInto::<char>::try_into(match x {
                0x00..=0x20 => y + 0x0100, // 0x0100..=0x0120
                0x21..=0x7E => y,          // 0x0021..=0x007E
                0x7F..=0xA0 => y + 0x00A2, // 0x0121..=0x0142
                0xA1..=0xAC => y,          // 0x00A1..=0x00AC
                0xAD => 0x0143,            // 0x0143 (0xAD is SOFT HYPHEN)
                0xAE..=0xFF => y,          // 0x00AE..=0x00FF
            })
            .unwrap()
        })
        .collect()
}

// Tokenize `input` with the BPE algorithm.
pub fn tokenize(
    token_to_id: &HashMap<String, usize>,
    ranks: &HashMap<(String, String), u32>,
    model: &Model,
    input: &str,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let raw_tokens: Vec<String> = input
        .to_lowercase()
        .split(char::is_whitespace)
        .map(encode_unique_encoding)
        .collect();
    let ids = {
        let mut ids = Vec::new();
        ids.push(token_to_id["<|startoftext|>"]);
        for raw_token in raw_tokens.iter() {
            if token_to_id.contains_key(&format!("{raw_token}</w>")) {
                // `raw_token` is already a valid token.
                ids.push(token_to_id[&format!("{raw_token}</w>")]);
            } else {
                // `raw_token` is not a valid token.
                // Split `raw_token` and get valid tokens with the merge algorithm.
                let mut tokens: Vec<String> = raw_token.chars().map(|x| x.to_string()).collect();
                let len = tokens.len();
                tokens[len - 1] = format!("{}</w>", tokens[len - 1]);
                while tokens.len() >= 2 {
                    let pairs = {
                        let mut pairs = Vec::with_capacity(tokens.len() - 1);
                        for i_char in 0..tokens.len() - 1 {
                            let token0 = tokens[i_char].clone();
                            let token1 = tokens[i_char + 1].clone();
                            pairs.push((token0, token1));
                        }
                        pairs
                    };
                    let mut best_rank = u32::MAX;
                    let mut best_i_pair = usize::MAX;
                    for (i_pair, pair) in pairs.iter().enumerate() {
                        if ranks.contains_key(pair) && ranks[pair] < best_rank {
                            best_rank = ranks[pair];
                            best_i_pair = i_pair;
                        }
                    }
                    if best_i_pair == usize::MAX {
                        break;
                    }

                    tokens[best_i_pair] =
                        format!("{}{}", tokens[best_i_pair], tokens[best_i_pair + 1]);
                    tokens.remove(best_i_pair + 1);
                }

                for token in tokens {
                    ids.push(token_to_id[&token]);
                }
            }
        }
        let len = ids.len();
        ids.extend(vec![token_to_id["<|endoftext|>"]; model.n_ctx - len - 1]);
        ids.push(token_to_id["<|endoftext|>"]);
        ids
    };
    Ok(ids)
}
