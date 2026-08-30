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

use serde_json::Value;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};

pub struct Layer {
    pub g1: TypedTensor<f32>,
    pub t1: TypedTensor<f32>,
    pub w11: TypedTensor<f32>,
    pub b11: TypedTensor<f32>,
    pub w12: TypedTensor<f32>,
    pub b12: TypedTensor<f32>,
    pub g2: TypedTensor<f32>,
    pub t2: TypedTensor<f32>,
    pub w21: TypedTensor<f32>,
    pub b21: TypedTensor<f32>,
    pub w22: TypedTensor<f32>,
    pub b22: TypedTensor<f32>,
}

pub struct Model {
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    pub e: TypedTensor<f32>,
    pub c0: TypedTensor<f32>,
    pub c1: TypedTensor<f32>,
    pub c2: TypedTensor<f32>,
    pub c3: TypedTensor<f32>,
    pub id_embd_vecs: Vec<TypedTensor<f32>>,
    pub pos_embd_vecs: Vec<TypedTensor<f32>>,
    pub layers: Vec<Layer>,
    pub gf: TypedTensor<f32>,
    pub tf: TypedTensor<f32>,
}

fn validate_shape(tensor: &TypedTensor<f32>, expected: [usize; 2]) -> Result<(), Box<dyn Error>> {
    let shape = tensor.shape();
    if shape != expected {
        let message = format!("shape of tensor {shape:?} differs from expected {expected:?}");
        return Err(message.into());
    }
    Ok(())
}

pub fn get_model(
    tensors: HashMap<String, TypedTensor<f32>>,
    config: HashMap<String, Value>,
) -> Result<Model, Box<dyn Error>> {
    let n_ctx = config["max_position_embeddings"].as_u64().unwrap() as usize;
    let n_embd = config["hidden_size"].as_u64().unwrap() as usize;
    let n_head = config["num_attention_heads"].as_u64().unwrap() as usize;
    let n_layer = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;

    let e = {
        let value = config["layer_norm_eps"].as_f64().unwrap() as f32;
        TypedTensor::<f32>::from_vec_col_major(vec![], vec![value])?
    };
    let c0 = {
        let value = 1.0f32 / n_embd as f32;
        TypedTensor::<f32>::from_vec_col_major(vec![], vec![value])?
    };
    let c1 = {
        let value = 1.0f32 / ((n_embd / n_head) as f32).sqrt();
        TypedTensor::<f32>::from_vec_col_major(vec![], vec![value])?
    };
    let c2 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![-1.702f32])?;
    let c3 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![1.0f32])?;

    let mut backend = CpuBackend::new();
    let prefix = "cond_stage_model.transformer.text_model";
    let id_embd_vecs = {
        let wte = &tensors[&format!("{prefix}.embeddings.token_embedding.weight")];
        let mut embd_vecs = Vec::with_capacity(vocab_size);
        for row in 0..vocab_size {
            let mut colmaj = Vec::with_capacity(n_embd);
            for col in 0..n_embd {
                colmaj.push(*wte.get(&[row, col])?);
            }
            let embd_vec = TypedTensor::<f32>::from_vec_col_major(vec![n_embd, 1], colmaj)?;
            embd_vecs.push(embd_vec);
        }
        embd_vecs
    };
    let pos_embd_vecs = {
        let wpe = &tensors[&format!("{prefix}.embeddings.position_embedding.weight")];
        let mut embd_vecs = Vec::with_capacity(n_ctx);
        for row in 0..n_ctx {
            let mut colmaj = Vec::with_capacity(n_embd);
            for col in 0..n_embd {
                colmaj.push(*wpe.get(&[row, col])?);
            }
            let embd_vec = TypedTensor::<f32>::from_vec_col_major(vec![n_embd, 1], colmaj)?;
            embd_vecs.push(embd_vec);
        }
        embd_vecs
    };
    let layers = {
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let g1 = tensors[&format!("{prefix}.encoder.layers.{i}.layer_norm1.weight")]
                .reshape(&[n_embd, 1], &mut backend)?;
            let t1 = tensors[&format!("{prefix}.encoder.layers.{i}.layer_norm1.bias")]
                .reshape(&[n_embd, 1], &mut backend)?;
            let w11 = {
                let w11q =
                    &tensors[&format!("{prefix}.encoder.layers.{i}.self_attn.q_proj.weight")];
                let w11k =
                    &tensors[&format!("{prefix}.encoder.layers.{i}.self_attn.k_proj.weight")];
                let w11v =
                    &tensors[&format!("{prefix}.encoder.layers.{i}.self_attn.v_proj.weight")];
                let mut colmaj = Vec::with_capacity(3 * n_embd * n_embd);
                for col in 0..n_embd {
                    for row in 0..n_embd {
                        colmaj.push(*w11q.get(&[row, col])?);
                    }
                    for row in 0..n_embd {
                        colmaj.push(*w11k.get(&[row, col])?);
                    }
                    for row in 0..n_embd {
                        colmaj.push(*w11v.get(&[row, col])?);
                    }
                }
                TypedTensor::<f32>::from_vec_col_major(vec![n_embd * 3, n_embd], colmaj)?
            };
            let b11 = {
                let b11q = &tensors[&format!("{prefix}.encoder.layers.{i}.self_attn.q_proj.bias")];
                let b11k = &tensors[&format!("{prefix}.encoder.layers.{i}.self_attn.k_proj.bias")];
                let b11v = &tensors[&format!("{prefix}.encoder.layers.{i}.self_attn.v_proj.bias")];
                let mut colmaj = Vec::with_capacity(3 * n_embd);
                for row in 0..n_embd {
                    colmaj.push(*b11q.get(&[row])?);
                }
                for row in 0..n_embd {
                    colmaj.push(*b11k.get(&[row])?);
                }
                for row in 0..n_embd {
                    colmaj.push(*b11v.get(&[row])?);
                }
                TypedTensor::<f32>::from_vec_col_major(vec![n_embd * 3, 1], colmaj)?
            };
            let w12 = tensors[&format!("{prefix}.encoder.layers.{i}.self_attn.out_proj.weight")]
                .duplicate()?;
            validate_shape(&w12, [n_embd, n_embd])?;
            let b12 = tensors[&format!("{prefix}.encoder.layers.{i}.self_attn.out_proj.bias")]
                .reshape(&[n_embd, 1], &mut backend)?;
            let g2 = tensors[&format!("{prefix}.encoder.layers.{i}.layer_norm2.weight")]
                .reshape(&[n_embd, 1], &mut backend)?;
            let t2 = tensors[&format!("{prefix}.encoder.layers.{i}.layer_norm2.bias")]
                .reshape(&[n_embd, 1], &mut backend)?;
            let w21 =
                tensors[&format!("{prefix}.encoder.layers.{i}.mlp.fc1.weight")].duplicate()?;
            validate_shape(&w21, [n_embd * 4, n_embd])?;
            let b21 = tensors[&format!("{prefix}.encoder.layers.{i}.mlp.fc1.bias")]
                .reshape(&[n_embd * 4, 1], &mut backend)?;
            let w22 =
                tensors[&format!("{prefix}.encoder.layers.{i}.mlp.fc2.weight")].duplicate()?;
            validate_shape(&w22, [n_embd, n_embd * 4])?;
            let b22 = tensors[&format!("{prefix}.encoder.layers.{i}.mlp.fc2.bias")]
                .reshape(&[n_embd, 1], &mut backend)?;
            let layer = Layer {
                g1,
                t1,
                w11,
                b11,
                w12,
                b12,
                g2,
                t2,
                w21,
                b21,
                w22,
                b22,
            };
            layers.push(layer);
        }
        layers
    };
    let gf = tensors[&format!("{prefix}.final_layer_norm.weight")]
        .reshape(&[n_embd, 1], &mut backend)?;
    let tf =
        tensors[&format!("{prefix}.final_layer_norm.bias")].reshape(&[n_embd, 1], &mut backend)?;

    let model = Model {
        n_ctx,
        n_embd,
        n_head,
        n_layer,
        vocab_size,
        e,
        c0,
        c1,
        c2,
        c3,
        id_embd_vecs,
        pos_embd_vecs,
        layers,
        gf,
        tf,
    };
    Ok(model)
}
