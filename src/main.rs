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
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde_json::Value;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::TypedTensor;

pub mod loader;
pub mod model;
pub mod tokenizer;
pub mod transformer;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        println!("GPT-2 Inference with tenferro");
        println!(
            "Usage: \x1b[1m{} <path to model repository> <your prompt>\x1b[22m",
            &args[0]
        );
        println!("You may have to enclose 'your prompt' with quotes.");
        return Ok(());
    }

    let token_to_id: HashMap<String, usize> = {
        let path = &format!("{}/vocab.json", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)?
    };

    let ranks = {
        let path = &format!("{}/merges.txt", &args[1]);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut ranks = HashMap::new();
        let mut rank = 0u32;
        for line in reader.lines().map_while(Result::ok) {
            // Skip a comment line.
            if line.starts_with("#") {
                continue;
            }
            let mut split = line.split(" ");
            let token0 = split.next().unwrap().to_string();
            let token1 = split.next().unwrap().to_string();
            ranks.insert((token0, token1), rank);
            rank += 1;
        }
        ranks
    };

    let model = {
        let tensors = {
            let path = &format!("{}/model.safetensors", &args[1]);
            loader::load_safetensors(path)?
        };
        let config: HashMap<String, Value> = {
            let path = &format!("{}/config.json", &args[1]);
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            serde_json::from_reader(reader)?
        };
        model::get_model(tensors, config)?
    };

    // ==== Tokenization ====

    // Token IDs
    let ids = tokenizer::tokenize(&token_to_id, &ranks, &model, &args[2])?;

    // ==== Inference ====

    let mut k_colmaj_caches = vec![Vec::<f32>::new(); model.n_layer];
    let mut v_colmaj_caches = vec![Vec::<f32>::new(); model.n_layer];
    let mut backend = CpuBackend::new();

    let x = {
        // I can utilize tenferro using col-major!
        // Adding a new column on the right is
        // equivalent to extending to colmaj.
        let mut colmaj = Vec::with_capacity(model.n_embd * model.n_ctx);
        for (pos, id) in ids.iter().enumerate() {
            let output = transformer::transformer(
                *id,
                pos,
                &model,
                &mut k_colmaj_caches,
                &mut v_colmaj_caches,
                &mut backend,
            )?;
            colmaj.extend_from_slice(output.host_data()?);
        }
        TypedTensor::<f32>::from_vec_col_major(vec![model.n_embd, model.n_ctx], colmaj)?
    };

    transformer::show(&x)?;

    Ok(())
}
