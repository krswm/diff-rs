// My U-net implementation with tenferro!

use std::error::Error;
use std::io::{Write, stdout};

use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

use crate::model::{Fablock, Fmodel, Frblock, Frcblock};
use crate::util::{conv11, conv31, cross_attention, gelu, groupnorm, groupnorm_micro, layernorm, self_attention, show, silu};

fn calc_frblock(
    tensor: &TypedTensor<f32>,
    timef: &TypedTensor<f32>,
    frblock: &Frblock,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // tensor [x, y, c, n]

    let num_c = tensor.shape()[2];

    let tmp = groupnorm(tensor, &frblock.g1, &frblock.t1, 32, backend)?;
    let tmp = silu(&tmp, backend)?;
    let tmp = conv31(&tmp, &frblock.wc1, &frblock.bc1, backend)?;

    let timef = silu(&timef, backend)?;
    let timef = timef.reshape(&[1280, 1], backend)?;
    let b = frblock.b.reshape(&[num_c, 1], backend)?;
    let timef = frblock.w.matmul(&timef, backend)?.add(&b, backend)?;

    let tmp = tmp.add(&timef, backend)?;
    let tmp = groupnorm(&tmp, &frblock.g2, &frblock.t2, 32, backend)?;
    let tmp = silu(&tmp, backend)?;
    let tmp = conv31(&tmp, &frblock.wc2, &frblock.bc2, backend)?;

    let tensor = tensor.add(&tmp, backend)?;

    Ok(tensor)
}

fn calc_frcblock(
    tensor: &TypedTensor<f32>,
    timef: &TypedTensor<f32>,
    frcblock: &Frcblock,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // tensor [x, y, c, n]

    let num_c = tensor.shape()[2];

    let tmp = groupnorm(tensor, &frcblock.g1, &frcblock.t1, 32, backend)?;
    let tmp = silu(&tmp, backend)?;
    let tmp = conv31(&tmp, &frcblock.wc1, &frcblock.bc1, backend)?;

    let timef = silu(&timef, backend)?;
    let timef = timef.reshape(&[1280, 1], backend)?;
    let b = frcblock.b.reshape(&[num_c, 1], backend)?;
    let timef = frcblock.w.matmul(&timef, backend)?.add(&b, backend)?;

    let tmp = tmp.add(&timef, backend)?;
    let tmp = groupnorm(&tmp, &frcblock.g2, &frcblock.t2, 32, backend)?;
    let tmp = silu(&tmp, backend)?;
    let tmp = conv31(&tmp, &frcblock.wc2, &frcblock.bc2, backend)?;

    let tensor = conv11(&tensor, &frcblock.wc3, &frcblock.bc3, backend)?;
    let tensor = tensor.add(&tmp, backend)?;

    Ok(tensor)
}

fn calc_fablock(
    tensor: &TypedTensor<f32>,
    context: &TypedTensor<f32>,
    fablock: &Fablock,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    println!("\x1b[91m");
    show(&tensor)?;

    let tmp = groupnorm_micro(tensor, &fablock.g1, &fablock.t1, 32, backend)?;
    // TODO: I just noticed the reference implementation has eps=1e-6 here, not usual 1e-5.
    // I totally missed that in my Julia version.
    // This may be the true cause of the slight calculation error in my Julia version!
    // I believe it won't affect the result image drastically, though.

    println!("\x1b[92m");
    show(&tmp)?;

    let tmp = conv11(&tmp, &fablock.wc1, &fablock.bc1, backend)?;

    println!("\x1b[93m");
    show(&tmp)?;

    let tmp2 = layernorm(&tmp, &fablock.g2, &fablock.t2, backend)?;

    println!("\x1b[94m");
    show(&tmp2)?;

    let num_c1 = fablock.w21.shape()[0];
    let zero_vector = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.0])?;
    let zero_vector = zero_vector.broadcast_in_dim(&[num_c1], &[], backend)?;
    let tmp2 = self_attention(&tmp2, &fablock.w21, &zero_vector, &fablock.w22, &fablock.b22, 8, backend)?;

    println!("\x1b[95m");
    show(&tmp2)?;

    let tmp = tmp.add(&tmp2, backend)?;

    println!("\x1b[96m");
    show(&tensor)?;

    let tmp2 = layernorm(&tmp, &fablock.g3, &fablock.t3, backend)?;

    println!("\x1b[31m");
    show(&tmp2)?;

    let tmp2 = cross_attention(&tmp2, &context, &fablock.w31q, &fablock.w31k, &fablock.w31v, &fablock.w32, &fablock.b32, 8, backend)?;

    println!("\x1b[32m");
    show(&tmp2)?;

    let tmp = tmp.add(&tmp2, backend)?;

    println!("\x1b[33m");
    show(&tmp2)?;

    let tmp2 = layernorm(&tmp, &fablock.g4, &fablock.t4, backend)?; // [x, y, c, n]

    println!("\x1b[34m");
    show(&tmp2)?;

    let num_d = fablock.b41.shape()[0];
    let b41 = fablock.b41.reshape(&[1, 1, 1, num_d], backend)?; // [1, 1, 1, d]
    let tmp2 = [&fablock.w41, &tmp2].einsum("dc,xycn->xynd", backend)? // [x, y, n, d]
        .add(&b41, backend)?; // [x, y, n, d]

    // chunk split for dimension d into half
    let num_x = tmp2.shape()[0];
    let num_y = tmp2.shape()[1];
    let num_n = tmp2.shape()[2];
    let num_e = num_d / 2;
    let mut chunks = tmp2.host_data()?.chunks(num_x * num_y * num_n * num_e); // as if [2][x, y, n, e]

    let tmp2 = TypedTensor::<f32>::from_vec_col_major(
        vec![num_x, num_y, num_n, num_e],
        chunks.next().unwrap().to_vec(),
    )?; // [x, y, n, e]
    let tmp3 = TypedTensor::<f32>::from_vec_col_major(
        vec![num_x, num_y, num_n, num_e],
        chunks.next().unwrap().to_vec(),
    )?; // [x, y, n, e]

    println!("\x1b[35m");
    show(&tmp2)?;

    println!("\x1b[35m");
    show(&tmp3)?;

    let tmp3 = gelu(&tmp3, backend)?; // [x, y, n, e]
    let tmp2 = tmp2.mul(&tmp3, backend)?; // [x, y, n, e]

    println!("\x1b[36m");
    show(&tmp2)?;

    let num_c = fablock.b42.shape()[0];
    let b42 = fablock.b42.reshape(&[1, 1, num_c, 1], backend)?; // [1, 1, c, 1]
    let tmp2 = [&fablock.w42, &tmp2].einsum("ce,xyne->xycn", backend)? // [x, y, c, n]
        .add(&b42, backend)?; // [x, y, c, n]

    println!("\x1b[91m");
    show(&tmp2)?;

    let tmp = tmp.add(&tmp2, backend)?;

    println!("\x1b[92m");
    show(&tmp)?;

    let tmp = conv11(&tmp, &fablock.wc4, &fablock.bc4, backend)?;

    println!("\x1b[93m");
    show(&tmp)?;

    let tensor = tensor.add(&tmp, backend)?;

    println!("\x1b[94m");
    show(&tensor)?;

    Ok(tensor)
    // attention block finished!
}

pub fn forward(
    tensor: &TypedTensor<f32>,
    context: &TypedTensor<f32>,
    curr_time: i32,
    prev_time: i32,
    fmodel: &Fmodel,
) -> Result<(), Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    println!("--- t = {curr_time} (t_prev = {prev_time}) ----");

    let myriad = TypedTensor::<f32>::from_vec_col_major(vec![], vec![10000.0])?;
    let a_hundred_and_sixty = TypedTensor::<f32>::from_vec_col_major(vec![], vec![160.0])?;
    let curr_time_as_tensor = TypedTensor::<f32>::from_vec_col_major(vec![], vec![curr_time as f32])?;
    let a: Vec<f32> = (0..160).map(|b| b as f32).collect();
    let timef = TypedTensor::<f32>::from_vec_col_major(vec![160], a)?;
    let timef = timef.div(&a_hundred_and_sixty, &mut backend)?.neg(&mut backend)?;
    let timef = myriad.pow(&timef, &mut backend)?.mul(&curr_time_as_tensor, &mut backend)?;
    let cos_timef = timef.cos(&mut backend)?;
    let sin_timef = timef.sin(&mut backend)?;
    let mut colmaj = Vec::with_capacity(320);
    colmaj.extend_from_slice(cos_timef.host_data()?);
    colmaj.extend_from_slice(sin_timef.host_data()?);
    // Tensor, EagerTensor, and TracedTensor have `stack`... I may have to use them instead...
    let timef = TypedTensor::<f32>::from_vec_col_major(vec![320], colmaj)?;

    let tensor = tensor.broadcast_in_dim(&[64, 64, 4, 2], &[0, 1, 2, 3], &mut backend)?;

    let timef = timef.reshape(&[320, 1], &mut backend)?;
    let time_b1 = fmodel.time_b1.reshape(&[1280, 1], &mut backend)?;
    let timef = fmodel.time_w1.matmul(&timef, &mut backend)?.add(&time_b1, &mut backend)?;
    let timef = silu(&timef, &mut backend)?;
    let timef = timef.reshape(&[1280, 1], &mut backend)?;
    let time_b2 = fmodel.time_b2.reshape(&[1280, 1], &mut backend)?;
    let timef = fmodel.time_w2.matmul(&timef, &mut backend)?.add(&time_b2, &mut backend)?;

    print!("\rfconv_i0\x1b[K");   stdout().flush()?; let tensor = conv31(&tensor, &fmodel.fconv_i0.wc, &fmodel.fconv_i0.bc, &mut backend)?;
    print!("\rfrblock_i1\x1b[K"); stdout().flush()?; let tensor = calc_frblock(&tensor, &timef, &fmodel.frblock_i1, &mut backend)?;
    print!("\rfablock_i1\x1b[K"); stdout().flush()?; let tensor = calc_fablock(&tensor, &context, &fmodel.fablock_i1, &mut backend)?;
    println!("\r\x1b[K"); stdout().flush();

    Ok(())
}
