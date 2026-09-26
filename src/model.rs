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
        // .transpose(&[3, 2, 1, 0], &mut backend)? // [c, C, 1, 1]
        .reshape(&[num_c, num_c], &mut backend)?; // [c, C]

    let b2 = tensors[&format!("{prefix}.proj_out.bias")].duplicate()?;

    let x = Dablock { g, t, w1, b1, w2, b2 };
    Ok(x)
}

// Decoder model
pub struct Dmodel {
    pub dconv_pq: Dconv,
    pub dconv_in: Dconv,

    pub drblock_mid1: Drblock,
    pub dablock: Dablock,
    pub drblock_mid2: Drblock,

    pub drblock_30: Drblock,
    pub drblock_31: Drblock,
    pub drblock_32: Drblock,
    pub dconv_3: Dconv,

    pub drblock_20: Drblock,
    pub drblock_21: Drblock,
    pub drblock_22: Drblock,
    pub dconv_2: Dconv,

    pub drcblock_10: Drcblock,
    pub drblock_11: Drblock,
    pub drblock_12: Drblock,
    pub dconv_1: Dconv,

    pub drcblock_00: Drcblock,
    pub drblock_01: Drblock,
    pub drblock_02: Drblock,

    pub g: TypedTensor<f32>, // [c]
    pub t: TypedTensor<f32>, // [c]
    pub dconv_out: Dconv,
}

pub fn get_dmodel(tensors: HashMap<String, TypedTensor<f32>>) -> Result<Dmodel, Box<dyn Error>> {
    let dconv_pq = get_dconv(&tensors, "first_stage_model.post_quant_conv")?;
    let dconv_in = get_dconv(&tensors, "first_stage_model.decoder.conv_in")?;

    let drblock_mid1 = get_drblock(&tensors, "first_stage_model.decoder.mid.block_1")?;
    let dablock = get_dablock(&tensors, "first_stage_model.decoder.mid.attn_1")?;
    let drblock_mid2 = get_drblock(&tensors, "first_stage_model.decoder.mid.block_2")?;

    let drblock_30 = get_drblock(&tensors, "first_stage_model.decoder.up.3.block.0")?;
    let drblock_31 = get_drblock(&tensors, "first_stage_model.decoder.up.3.block.1")?;
    let drblock_32 = get_drblock(&tensors, "first_stage_model.decoder.up.3.block.2")?;
    let dconv_3 = get_dconv(&tensors, "first_stage_model.decoder.up.3.upsample.conv")?;

    let drblock_20 = get_drblock(&tensors, "first_stage_model.decoder.up.2.block.0")?;
    let drblock_21 = get_drblock(&tensors, "first_stage_model.decoder.up.2.block.1")?;
    let drblock_22 = get_drblock(&tensors, "first_stage_model.decoder.up.2.block.2")?;
    let dconv_2 = get_dconv(&tensors, "first_stage_model.decoder.up.2.upsample.conv")?;

    let drcblock_10 = get_drcblock(&tensors, "first_stage_model.decoder.up.1.block.0")?;
    let drblock_11 = get_drblock(&tensors, "first_stage_model.decoder.up.1.block.1")?;
    let drblock_12 = get_drblock(&tensors, "first_stage_model.decoder.up.1.block.2")?;
    let dconv_1 = get_dconv(&tensors, "first_stage_model.decoder.up.1.upsample.conv")?;

    let drcblock_00 = get_drcblock(&tensors, "first_stage_model.decoder.up.0.block.0")?;
    let drblock_01 = get_drblock(&tensors, "first_stage_model.decoder.up.0.block.1")?;
    let drblock_02 = get_drblock(&tensors, "first_stage_model.decoder.up.0.block.2")?;

    let g = tensors["first_stage_model.decoder.norm_out.weight"].duplicate()?;
    let t = tensors["first_stage_model.decoder.norm_out.bias"].duplicate()?;
    let dconv_out = get_dconv(&tensors, "first_stage_model.decoder.conv_out")?;

    let x = Dmodel {
        dconv_pq,
        dconv_in,

        drblock_mid1,
        dablock,
        drblock_mid2,

        drblock_30,
        drblock_31,
        drblock_32,
        dconv_3,

        drblock_20,
        drblock_21,
        drblock_22,
        dconv_2,

        drcblock_10,
        drblock_11,
        drblock_12,
        dconv_1,

        drcblock_00,
        drblock_01,
        drblock_02,

        g,
        t,
        dconv_out,
    };
    Ok(x)
}

// Diffusion convolution
pub struct Fconv {
    pub wc: TypedTensor<f32>,  // [x, y, i, o]
    pub bc: TypedTensor<f32>,  // [o]
}

pub fn get_fconv(
    tensors: &HashMap<String, TypedTensor<f32>>,
    prefix: &str,
) -> Result<Fconv, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let wc = tensors[&format!("{prefix}.weight")].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc = tensors[&format!("{prefix}.bias")].duplicate()?;

    let x = Fconv { wc, bc };

    Ok(x)
}

// Diffusion residual block
pub struct Frblock {
    pub g1: TypedTensor<f32>,
    pub t1: TypedTensor<f32>,
    pub wc1: TypedTensor<f32>,
    pub bc1: TypedTensor<f32>,
    pub w: TypedTensor<f32>,
    pub b: TypedTensor<f32>,
    pub g2: TypedTensor<f32>,
    pub t2: TypedTensor<f32>,
    pub wc2: TypedTensor<f32>,
    pub bc2: TypedTensor<f32>,
}

pub fn get_frblock(
    tensors: &HashMap<String, TypedTensor<f32>>,
    prefix: &str,
) -> Result<Frblock, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let g1  = tensors[&format!("{prefix}.in_layers.0.weight")].duplicate()?;
    let t1  = tensors[&format!("{prefix}.in_layers.0.bias")].duplicate()?;
    let wc1 = tensors[&format!("{prefix}.in_layers.2.weight")].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc1 = tensors[&format!("{prefix}.in_layers.2.bias")].duplicate()?;
    let w   = tensors[&format!("{prefix}.emb_layers.1.weight")].duplicate()?;
    let b   = tensors[&format!("{prefix}.emb_layers.1.bias")].duplicate()?;
    let g2  = tensors[&format!("{prefix}.out_layers.0.weight")].duplicate()?;
    let t2  = tensors[&format!("{prefix}.out_layers.0.bias")].duplicate()?;
    let wc2 = tensors[&format!("{prefix}.out_layers.3.weight")].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc2 = tensors[&format!("{prefix}.out_layers.3.bias")].duplicate()?;

    let x = Frblock { g1, t1, wc1, bc1, w, b, g2, t2, wc2, bc2 };
    Ok(x)
}

// Diffusion residual block w/ additional convolution
pub struct Frcblock {
    pub g1: TypedTensor<f32>,
    pub t1: TypedTensor<f32>,
    pub wc1: TypedTensor<f32>,
    pub bc1: TypedTensor<f32>,
    pub w: TypedTensor<f32>,
    pub b: TypedTensor<f32>,
    pub g2: TypedTensor<f32>,
    pub t2: TypedTensor<f32>,
    pub wc2: TypedTensor<f32>,
    pub bc2: TypedTensor<f32>,
    pub wc3: TypedTensor<f32>,
    pub bc3: TypedTensor<f32>,
}

pub fn get_frcblock(
    tensors: &HashMap<String, TypedTensor<f32>>,
    prefix: &str,
) -> Result<Frcblock, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let g1  = tensors[&format!("{prefix}.in_layers.0.weight")].duplicate()?;
    let t1  = tensors[&format!("{prefix}.in_layers.0.bias")].duplicate()?;
    let wc1 = tensors[&format!("{prefix}.in_layers.2.weight")].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc1 = tensors[&format!("{prefix}.in_layers.2.bias")].duplicate()?;
    let w   = tensors[&format!("{prefix}.emb_layers.1.weight")].duplicate()?;
    let b   = tensors[&format!("{prefix}.emb_layers.1.bias")].duplicate()?;
    let g2  = tensors[&format!("{prefix}.out_layers.0.weight")].duplicate()?;
    let t2  = tensors[&format!("{prefix}.out_layers.0.bias")].duplicate()?;
    let wc2 = tensors[&format!("{prefix}.out_layers.3.weight")].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc2 = tensors[&format!("{prefix}.out_layers.3.bias")].duplicate()?;
    let wc3 = tensors[&format!("{prefix}.skip_connection.weight")].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc3 = tensors[&format!("{prefix}.skip_connection.bias")].duplicate()?;

    let x = Frcblock { g1, t1, wc1, bc1, w, b, g2, t2, wc2, bc2, wc3, bc3 };
    Ok(x)
}

// Diffusion attention block
pub struct Fablock {
    pub g1: TypedTensor<f32>,
    pub t1: TypedTensor<f32>,
    pub wc1: TypedTensor<f32>,
    pub bc1: TypedTensor<f32>,

    pub g2: TypedTensor<f32>,
    pub t2: TypedTensor<f32>,
    pub w21: TypedTensor<f32>,
    pub w22: TypedTensor<f32>,
    pub b22: TypedTensor<f32>,

    pub g3: TypedTensor<f32>,
    pub t3: TypedTensor<f32>,
    pub w31q: TypedTensor<f32>,
    pub w31k: TypedTensor<f32>,
    pub w31v: TypedTensor<f32>,
    pub w32: TypedTensor<f32>,
    pub b32: TypedTensor<f32>,

    pub g4: TypedTensor<f32>,
    pub t4: TypedTensor<f32>,
    pub w41: TypedTensor<f32>,
    pub b41: TypedTensor<f32>,
    pub w42: TypedTensor<f32>,
    pub b42: TypedTensor<f32>,
    pub wc4: TypedTensor<f32>,
    pub bc4: TypedTensor<f32>,
}

pub fn get_fablock(
    tensors: &HashMap<String, TypedTensor<f32>>,
    prefix: &str,
) -> Result<Fablock, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let g1   = tensors[&format!("{prefix}.norm.weight")].duplicate()?;
    let t1   = tensors[&format!("{prefix}.norm.bias")].duplicate()?;
    let wc1  = tensors[&format!("{prefix}.proj_in.weight")].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc1  = tensors[&format!("{prefix}.proj_in.bias")].duplicate()?;

    let g2   = tensors[&format!("{prefix}.transformer_blocks.0.norm1.weight")].duplicate()?;
    let t2   = tensors[&format!("{prefix}.transformer_blocks.0.norm1.bias")].duplicate()?;

    let w21 = {
        let mut colmaj = Vec::new();
        colmaj.extend_from_slice(tensors[&format!("{prefix}.transformer_blocks.0.attn1.to_q.weight")].transpose(&[1, 0], &mut backend)?.host_data()?);
        colmaj.extend_from_slice(tensors[&format!("{prefix}.transformer_blocks.0.attn1.to_k.weight")].transpose(&[1, 0], &mut backend)?.host_data()?);
        colmaj.extend_from_slice(tensors[&format!("{prefix}.transformer_blocks.0.attn1.to_v.weight")].transpose(&[1, 0], &mut backend)?.host_data()?);

        let a = tensors[&format!("{prefix}.transformer_blocks.0.attn1.to_q.weight")].shape()[0];

        TypedTensor::<f32>::from_vec_col_major(vec![a, a * 3], colmaj)?.transpose(&[1, 0], &mut backend)?
    };

    // No bias (b21). Use zero vector.
    let w22  = tensors[&format!("{prefix}.transformer_blocks.0.attn1.to_out.0.weight")].duplicate()?;
    let b22  = tensors[&format!("{prefix}.transformer_blocks.0.attn1.to_out.0.bias")].duplicate()?;

    let g3   = tensors[&format!("{prefix}.transformer_blocks.0.norm2.weight")].duplicate()?;
    let t3   = tensors[&format!("{prefix}.transformer_blocks.0.norm2.bias")].duplicate()?;
    let w31q = tensors[&format!("{prefix}.transformer_blocks.0.attn2.to_q.weight")].duplicate()?;
    let w31k = tensors[&format!("{prefix}.transformer_blocks.0.attn2.to_k.weight")].duplicate()?;
    let w31v = tensors[&format!("{prefix}.transformer_blocks.0.attn2.to_v.weight")].duplicate()?;
    let w32  = tensors[&format!("{prefix}.transformer_blocks.0.attn2.to_out.0.weight" )].duplicate()?;
    let b32  = tensors[&format!("{prefix}.transformer_blocks.0.attn2.to_out.0.bias" )].duplicate()?;

    let g4   = tensors[&format!("{prefix}.transformer_blocks.0.norm3.weight")].duplicate()?;
    let t4   = tensors[&format!("{prefix}.transformer_blocks.0.norm3.bias")].duplicate()?;
    let w41  = tensors[&format!("{prefix}.transformer_blocks.0.ff.net.0.proj.weight")].transpose(&[1, 0], &mut backend)?;
    let b41  = tensors[&format!("{prefix}.transformer_blocks.0.ff.net.0.proj.bias")].duplicate()?;
    let w42  = tensors[&format!("{prefix}.transformer_blocks.0.ff.net.2.weight")].transpose(&[1, 0], &mut backend)?;
    let b42  = tensors[&format!("{prefix}.transformer_blocks.0.ff.net.2.bias")].duplicate()?;
    let wc4  = tensors[&format!("{prefix}.proj_out.weight")].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc4  = tensors[&format!("{prefix}.proj_out.bias")].duplicate()?;

    let x = Fablock {
        g1, t1, wc1, bc1,
        g2, t2, w21, w22, b22,
        g3, t3, w31q, w31k, w31v, w32, b32,
        g4, t4, w41, b41, w42, b42, wc4, bc4,
    };
    Ok(x)
}

// diffusion model
pub struct Fmodel {
    pub time_w1: TypedTensor<f32>,
    pub time_b1: TypedTensor<f32>,
    pub time_w2: TypedTensor<f32>,
    pub time_b2: TypedTensor<f32>,

    pub fconv_i0: Fconv,
    pub fconv_i3: Fconv,
    pub fconv_i6: Fconv,
    pub fconv_i9: Fconv,
    pub fconv_o2: Fconv,
    pub fconv_o5: Fconv,
    pub fconv_o8: Fconv,

    pub frblock_i1: Frblock,
    pub frblock_i2: Frblock,
    pub frblock_i5: Frblock,
    pub frblock_i8: Frblock,
    pub frblock_i10: Frblock,
    pub frblock_i11: Frblock,
    pub frblock_m0: Frblock,
    pub frblock_m2: Frblock,

    pub frcblock_i4: Frcblock,
    pub frcblock_i7: Frcblock,
    pub frcblock_o0: Frcblock,
    pub frcblock_o1: Frcblock,
    pub frcblock_o2: Frcblock,
    pub frcblock_o3: Frcblock,
    pub frcblock_o4: Frcblock,
    pub frcblock_o5: Frcblock,
    pub frcblock_o6: Frcblock,
    pub frcblock_o7: Frcblock,
    pub frcblock_o8: Frcblock,
    pub frcblock_o9: Frcblock,
    pub frcblock_o10: Frcblock,
    pub frcblock_o11: Frcblock,

    pub fablock_i1: Fablock,
    pub fablock_i2: Fablock,
    pub fablock_i4: Fablock,
    pub fablock_i5: Fablock,
    pub fablock_i7: Fablock,
    pub fablock_i8: Fablock,
    pub fablock_m1: Fablock,
    pub fablock_o3: Fablock,
    pub fablock_o4: Fablock,
    pub fablock_o5: Fablock,
    pub fablock_o6: Fablock,
    pub fablock_o7: Fablock,
    pub fablock_o8: Fablock,
    pub fablock_o9: Fablock,
    pub fablock_o10: Fablock,
    pub fablock_o11: Fablock,

    pub g_final: TypedTensor<f32>,
    pub t_final: TypedTensor<f32>,
    pub wc_final: TypedTensor<f32>,
    pub bc_final: TypedTensor<f32>,
}

pub fn get_fmodel(tensors: HashMap<String, TypedTensor<f32>>) -> Result<Fmodel, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let time_w1 = tensors["model.diffusion_model.time_embed.0.weight"].duplicate()?;
    let time_b1 = tensors["model.diffusion_model.time_embed.0.bias"].duplicate()?;
    let time_w2 = tensors["model.diffusion_model.time_embed.2.weight"].duplicate()?;
    let time_b2 = tensors["model.diffusion_model.time_embed.2.bias"].duplicate()?;

    let fconv_i0 = get_fconv(&tensors, "model.diffusion_model.input_blocks.0.0")?;
    let fconv_i3 = get_fconv(&tensors, "model.diffusion_model.input_blocks.3.0.op")?;
    let fconv_i6 = get_fconv(&tensors, "model.diffusion_model.input_blocks.6.0.op")?;
    let fconv_i9 = get_fconv(&tensors, "model.diffusion_model.input_blocks.9.0.op")?;
    let fconv_o2 = get_fconv(&tensors, "model.diffusion_model.output_blocks.2.1.conv")?;
    let fconv_o5 = get_fconv(&tensors, "model.diffusion_model.output_blocks.5.2.conv")?;
    let fconv_o8 = get_fconv(&tensors, "model.diffusion_model.output_blocks.8.2.conv")?;

    let frblock_i1  = get_frblock(&tensors, "model.diffusion_model.input_blocks.1.0")?;
    let frblock_i2  = get_frblock(&tensors, "model.diffusion_model.input_blocks.2.0")?;
    let frblock_i5  = get_frblock(&tensors, "model.diffusion_model.input_blocks.5.0")?;
    let frblock_i8  = get_frblock(&tensors, "model.diffusion_model.input_blocks.8.0")?;
    let frblock_i10 = get_frblock(&tensors, "model.diffusion_model.input_blocks.10.0")?;
    let frblock_i11 = get_frblock(&tensors, "model.diffusion_model.input_blocks.11.0")?;
    let frblock_m0  = get_frblock(&tensors, "model.diffusion_model.middle_block.0")?;
    let frblock_m2  = get_frblock(&tensors, "model.diffusion_model.middle_block.2")?;

    let frcblock_i4  = get_frcblock(&tensors, "model.diffusion_model.input_blocks.4.0")?;
    let frcblock_i7  = get_frcblock(&tensors, "model.diffusion_model.input_blocks.7.0")?;
    let frcblock_o0  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.0.0")?;
    let frcblock_o1  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.1.0")?;
    let frcblock_o2  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.2.0")?;
    let frcblock_o3  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.3.0")?;
    let frcblock_o4  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.4.0")?;
    let frcblock_o5  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.5.0")?;
    let frcblock_o6  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.6.0")?;
    let frcblock_o7  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.7.0")?;
    let frcblock_o8  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.8.0")?;
    let frcblock_o9  = get_frcblock(&tensors, "model.diffusion_model.output_blocks.9.0")?;
    let frcblock_o10 = get_frcblock(&tensors, "model.diffusion_model.output_blocks.10.0")?;
    let frcblock_o11 = get_frcblock(&tensors, "model.diffusion_model.output_blocks.11.0")?;

    let fablock_i1  = get_fablock(&tensors, "model.diffusion_model.input_blocks.1.1")?;
    let fablock_i2  = get_fablock(&tensors, "model.diffusion_model.input_blocks.2.1")?;
    let fablock_i4  = get_fablock(&tensors, "model.diffusion_model.input_blocks.4.1")?;
    let fablock_i5  = get_fablock(&tensors, "model.diffusion_model.input_blocks.5.1")?;
    let fablock_i7  = get_fablock(&tensors, "model.diffusion_model.input_blocks.7.1")?;
    let fablock_i8  = get_fablock(&tensors, "model.diffusion_model.input_blocks.8.1")?;
    let fablock_m1  = get_fablock(&tensors, "model.diffusion_model.middle_block.1")?;
    let fablock_o3  = get_fablock(&tensors, "model.diffusion_model.output_blocks.3.1")?;
    let fablock_o4  = get_fablock(&tensors, "model.diffusion_model.output_blocks.4.1")?;
    let fablock_o5  = get_fablock(&tensors, "model.diffusion_model.output_blocks.5.1")?;
    let fablock_o6  = get_fablock(&tensors, "model.diffusion_model.output_blocks.6.1")?;
    let fablock_o7  = get_fablock(&tensors, "model.diffusion_model.output_blocks.7.1")?;
    let fablock_o8  = get_fablock(&tensors, "model.diffusion_model.output_blocks.8.1")?;
    let fablock_o9  = get_fablock(&tensors, "model.diffusion_model.output_blocks.9.1")?;
    let fablock_o10 = get_fablock(&tensors, "model.diffusion_model.output_blocks.10.1")?;
    let fablock_o11 = get_fablock(&tensors, "model.diffusion_model.output_blocks.11.1")?;

    let g_final  = tensors["model.diffusion_model.out.0.weight"].duplicate()?;
    let t_final  = tensors["model.diffusion_model.out.0.bias"].duplicate()?;
    let wc_final = tensors["model.diffusion_model.out.2.weight"].transpose(&[3, 2, 1, 0], &mut backend)?;
    let bc_final = tensors["model.diffusion_model.out.2.bias"].duplicate()?;

    let x = Fmodel {
        time_w1,
        time_b1,
        time_w2,
        time_b2,

        fconv_i0,
        fconv_i3,
        fconv_i6,
        fconv_i9,
        fconv_o2,
        fconv_o5,
        fconv_o8,

        frblock_i1,  
        frblock_i2,  
        frblock_i5,  
        frblock_i8,  
        frblock_i10, 
        frblock_i11, 
        frblock_m0,  
        frblock_m2,

        frcblock_i4,  
        frcblock_i7,  
        frcblock_o0,  
        frcblock_o1,  
        frcblock_o2,  
        frcblock_o3,  
        frcblock_o4,  
        frcblock_o5,  
        frcblock_o6,  
        frcblock_o7,  
        frcblock_o8,  
        frcblock_o9,  
        frcblock_o10, 
        frcblock_o11,

        fablock_i1,
        fablock_i2,
        fablock_i4,
        fablock_i5,
        fablock_i7,
        fablock_i8,
        fablock_m1,
        fablock_o3,
        fablock_o4,
        fablock_o5,
        fablock_o6,
        fablock_o7,
        fablock_o8,
        fablock_o9,
        fablock_o10,
        fablock_o11,

        g_final,
        t_final,
        wc_final,
        bc_final,
    };
    Ok(x)
}
