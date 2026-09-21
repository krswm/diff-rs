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
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

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

////

// The model safetensor file seems to be constructed with
// row-major engine (PyTorch etc.) in mind
// (rightmost index varies the most, batch index is on the left).
// However, tenferro is a col-major tensor library
// (leftmost index varies the most, batch index is on the right).
// Therefore I'll `transpose` the tensors from the safetensors file.
//
// Note that in fact `loader.rs` converts row-major to col-major as well
// so what's I'm doing here is just reverting it.
// I have to re-consider about my implementation...

// decoder convolution
pub struct Dconv {
    pub wc: TypedTensor<f32>, // [x, y, i, o]
    pub bc: TypedTensor<f32>, // [o]
}

pub fn get_dconv(
    tensors: &HashMap<String, TypedTensor<f32>>,
    prefix: &str,
) -> Result<Dconv, Box<dyn Error>> {
    let mut backend = CpuBackend::new();
   
    let wc = tensors[&format!("{prefix}.weight")] // [o, i, y, x]
        .transpose(&[3, 2, 1, 0], &mut backend)?; // [x, y, i, o]
    let bc = tensors[&format!("{prefix}.bias")].duplicate()?; // [o]

    let x = Dconv { wc, bc };
    Ok(x)
}

// Decoder residual block
pub struct Drblock {
    pub g1: TypedTensor<f32>, // [c]
    pub t1: TypedTensor<f32>, // [c]
    pub wc1: TypedTensor<f32>, // [x, y, i, o]
    pub bc1: TypedTensor<f32>, // [o]
    pub g2: TypedTensor<f32>, // [c]
    pub t2: TypedTensor<f32>, // [c]
    pub wc2: TypedTensor<f32>, // [x, y, i, o]
    pub bc2: TypedTensor<f32>, // [o]
}

pub fn get_drblock(
    tensors: &HashMap<String, TypedTensor<f32>>,
    prefix: &str,
) -> Result<Drblock, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let g1 = tensors[&format!("{prefix}.norm1.weight")].duplicate()?;
    let t1 = tensors[&format!("{prefix}.norm1.bias")].duplicate()?;
    let wc1 = tensors[&format!("{prefix}.conv1.weight")] // [o, i, y, x]
        .transpose(&[3, 2, 1, 0], &mut backend)?; // [x, y, i, o]
    let bc1 = tensors[&format!("{prefix}.conv1.bias")].duplicate()?;
    let g2 = tensors[&format!("{prefix}.norm2.weight")].duplicate()?;
    let t2 = tensors[&format!("{prefix}.norm2.bias")].duplicate()?;
    let wc2 = tensors[&format!("{prefix}.conv2.weight")] // [o, i, y, x]
        .transpose(&[3, 2, 1, 0], &mut backend)?; // [x, y, i, o]
    let bc2 = tensors[&format!("{prefix}.conv2.bias")].duplicate()?;

    let x = Drblock { g1, t1, wc1, bc1, g2, t2, wc2, bc2 };
    Ok(x)
}

// Decoder residual block w/ additional convolution
pub struct Drcblock {
    pub g1: TypedTensor<f32>, // [c]
    pub t1: TypedTensor<f32>, // [c]
    pub wc1: TypedTensor<f32>, // [x, y, i, o]
    pub bc1: TypedTensor<f32>, // [o]
    pub g2: TypedTensor<f32>, // [c]
    pub t2: TypedTensor<f32>, // [c]
    pub wc2: TypedTensor<f32>, // [x, y, i, o]
    pub bc2: TypedTensor<f32>, // [o]
    pub wc3: TypedTensor<f32>, // [x, y, i, o]
    pub bc3: TypedTensor<f32>, // [o]
}

pub fn get_drcblock(
    tensors: &HashMap<String, TypedTensor<f32>>,
    prefix: &str,
) -> Result<Drcblock, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let g1 = tensors[&format!("{prefix}.norm1.weight")].duplicate()?;
    let t1 = tensors[&format!("{prefix}.norm1.bias")].duplicate()?;
    let wc1 = tensors[&format!("{prefix}.conv1.weight")] // [o, i, y, x]
        .transpose(&[3, 2, 1, 0], &mut backend)?; // [x, y, i, o]
    let bc1 = tensors[&format!("{prefix}.conv1.bias")].duplicate()?;
    let g2 = tensors[&format!("{prefix}.norm2.weight")].duplicate()?;
    let t2 = tensors[&format!("{prefix}.norm2.bias")].duplicate()?;
    let wc2 = tensors[&format!("{prefix}.conv2.weight")] // [o, i, y, x]
        .transpose(&[3, 2, 1, 0], &mut backend)?; // [x, y, i, o]
    let bc2 = tensors[&format!("{prefix}.conv2.bias")].duplicate()?;
    let wc3 = tensors[&format!("{prefix}.nin_shortcut.weight")] // [o, i, y, x]
        .transpose(&[3, 2, 1, 0], &mut backend)?; // [x, y, i, o]
    let bc3 = tensors[&format!("{prefix}.nin_shortcut.bias")].duplicate()?;

    let x = Drcblock { g1, t1, wc1, bc1, g2, t2, wc2, bc2, wc3, bc3 };
    Ok(x)
}

// Decoder attention block
pub struct Dablock {
    pub g: TypedTensor<f32>, // [c]
    pub t: TypedTensor<f32>, // [c]
    pub w1: TypedTensor<f32>, // [d, c]
    pub b1: TypedTensor<f32>, // [d]
    pub w2: TypedTensor<f32>, // [c, C]
    pub b2: TypedTensor<f32>, // [c]
}

pub fn get_dablock(
    tensors: &HashMap<String, TypedTensor<f32>>,
    prefix: &str,
) -> Result<Dablock, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let g = tensors[&format!("{prefix}.norm.weight")].duplicate()?;
    let t = tensors[&format!("{prefix}.norm.bias")].duplicate()?;

    //     +---+
    //     | Q |                T
    //     +---+   +---+---+---+ 
    // W = | K | = | Qᵀ| Kᵀ| Vᵀ|
    //     +---+   +---+---+---+
    //     | V |
    //     +---+

    let w1 = {
        let mut colmaj = Vec::new();
        colmaj.extend_from_slice(tensors[&format!("{prefix}.q.weight")].transpose(&[3, 2, 1, 0], &mut backend)?.host_data()?);
        colmaj.extend_from_slice(tensors[&format!("{prefix}.k.weight")].transpose(&[3, 2, 1, 0], &mut backend)?.host_data()?);
        colmaj.extend_from_slice(tensors[&format!("{prefix}.v.weight")].transpose(&[3, 2, 1, 0], &mut backend)?.host_data()?);

        let a = tensors[&format!("{prefix}.q.weight")].shape()[0];

        TypedTensor::<f32>::from_vec_col_major(vec![a, a * 3], colmaj)?.transpose(&[1, 0], &mut backend)?
    };

    let b1 = {
        let mut colmaj = Vec::new();
        colmaj.extend_from_slice(tensors[&format!("{prefix}.q.bias")].host_data()?);
        colmaj.extend_from_slice(tensors[&format!("{prefix}.k.bias")].host_data()?);
        colmaj.extend_from_slice(tensors[&format!("{prefix}.v.bias")].host_data()?);

        let a = tensors[&format!("{prefix}.q.bias")].shape()[0];

        TypedTensor::<f32>::from_vec_col_major(vec![a * 3], colmaj)?
    };

    let num_c = tensors[&format!("{prefix}.proj_out.weight")].shape()[0];

    let w2 = tensors[&format!("{prefix}.proj_out.weight")] // [C, c, 1, 1]
        .transpose(&[3, 2, 1, 0], &mut backend)? // [c, C, 1, 1]
        .reshape(&[num_c, num_c, 1, 1], &mut backend)?;

    let b2 = tensors[&format!("{prefix}.proj_out.bias")].duplicate()?;

    let x = Dablock { g, t, w1, b1, w2, b2 };
    Ok(x)
}

// Decoder model
pub struct Dmodel {
    pub dconv_pq: Dconv,
    pub drblock_mid1: Drblock,
    pub dablock: Dablock,
    pub drcblock_10: Drcblock,
}

pub fn get_dmodel(tensors: HashMap<String, TypedTensor<f32>>) -> Result<Dmodel, Box<dyn Error>> {
    let dconv_pq = get_dconv(&tensors, "first_stage_model.post_quant_conv")?;
    let drblock_mid1 = get_drblock(&tensors, "first_stage_model.decoder.mid.block_1")?;
    let dablock = get_dablock(&tensors, "first_stage_model.decoder.mid.attn_1")?;
    let drcblock_10 = get_drcblock(&tensors, "first_stage_model.decoder.up.1.block.0")?;

    let x = Dmodel {
        dconv_pq,
        drblock_mid1,
        dablock,
        drcblock_10,
    };
    Ok(x)
}
