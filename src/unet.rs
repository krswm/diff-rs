// My U-net implementation with tenferro!

use std::error::Error;
use std::io::{Write, stdout};

use rand::rngs::ChaCha20Rng;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

use crate::loader::load_safetensors;
use crate::model::{Fablock, Fconv, Fmodel, Frblock, Frcblock};
use crate::util::{conv11, conv31, conv32, cross_attention, gelu, groupnorm, groupnorm_micro, layernorm, randn, self_attention, show, silu, upsample};

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
    let num = frcblock.b.shape()[0];
    let b = frcblock.b.reshape(&[num, 1], backend)?;
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
    let tmp = groupnorm_micro(tensor, &fablock.g1, &fablock.t1, 32, backend)?;
    // TODO: I just noticed the reference implementation has eps=1e-6 here, not usual 1e-5.
    // I totally missed that in my Julia version.
    // This may be the true cause of the slight calculation error in my Julia version!
    // I believe it won't affect the result image drastically, though.

    let tmp = conv11(&tmp, &fablock.wc1, &fablock.bc1, backend)?;

    let tmp2 = layernorm(&tmp, &fablock.g2, &fablock.t2, backend)?;

    let num_c1 = fablock.w21.shape()[0];
    let zero_vector = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.0])?;
    let zero_vector = zero_vector.broadcast_in_dim(&[num_c1], &[], backend)?;
    let tmp2 = self_attention(&tmp2, &fablock.w21, &zero_vector, &fablock.w22, &fablock.b22, 8, backend)?;

    let tmp = tmp.add(&tmp2, backend)?;

    let tmp2 = layernorm(&tmp, &fablock.g3, &fablock.t3, backend)?;

    let tmp2 = cross_attention(&tmp2, &context, &fablock.w31q, &fablock.w31k, &fablock.w31v, &fablock.w32, &fablock.b32, 8, backend)?;

    let tmp = tmp.add(&tmp2, backend)?;

    let tmp2 = layernorm(&tmp, &fablock.g4, &fablock.t4, backend)?; // [x, y, c, n]

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

    let tmp3 = gelu(&tmp3, backend)?; // [x, y, n, e]
    let tmp2 = tmp2.mul(&tmp3, backend)?; // [x, y, n, e]

    let num_c = fablock.b42.shape()[0];
    let b42 = fablock.b42.reshape(&[1, 1, num_c, 1], backend)?; // [1, 1, c, 1]
    let tmp2 = [&fablock.w42, &tmp2].einsum("ce,xyne->xycn", backend)? // [x, y, c, n]
        .add(&b42, backend)?; // [x, y, c, n]

    let tmp = tmp.add(&tmp2, backend)?;

    let tmp = conv11(&tmp, &fablock.wc4, &fablock.bc4, backend)?;

    let tensor = tensor.add(&tmp, backend)?;

    Ok(tensor)
}

fn cat(
    tensor: &TypedTensor<f32>,
    tensor_ix: &TypedTensor<f32>,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // tensor    [x, y, c, n]
    // tensor_ix [x, y, c, n]

    // This function will concatenate `tensor` and `tensor_ix` along with the c-axis.
    // `tensor` first, then `tensor_ix`.

    let num_x = tensor.shape()[0];
    let num_y = tensor.shape()[1];
    let num_c0 = tensor.shape()[2];
    let num_c1 = tensor_ix.shape()[2];
    let num_n = tensor.shape()[3];

    let tensor = tensor.transpose(&[0, 1, 3, 2], backend)?; // [x, y, n, c]
    let tensor_ix = tensor_ix.transpose(&[0, 1, 3, 2], backend)?; // [x, y, n, c]
    let result = {
        let mut colmaj = Vec::new();
        colmaj.extend_from_slice(tensor.host_data()?);
        colmaj.extend_from_slice(tensor_ix.host_data()?);
        TypedTensor::<f32>::from_vec_col_major(vec![num_x, num_y, num_n, num_c0 + num_c1], colmaj)?
    }; // [x, y, n, c]
    let result = result.transpose(&[0, 1, 3, 2], backend)?; // [x, y, c, n]

    Ok(result)
}

fn calc_upsample(
    tensor: &TypedTensor<f32>,
    fconv: &Fconv,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let tensor = upsample(&tensor)?;
    let tensor = conv31(&tensor, &fconv.wc, &fconv.bc, backend)?;
    Ok(tensor)
}

pub fn forward(
    tensor_orig: &TypedTensor<f32>,
    context: &TypedTensor<f32>,
    curr_time: i32,
    prev_time: i32,
    fmodel: &Fmodel,
    rng: &mut ChaCha20Rng,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
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

    let tensor = tensor_orig.broadcast_in_dim(&[64, 64, 4, 2], &[0, 1, 2, 3], &mut backend)?;

    let timef = timef.reshape(&[320, 1], &mut backend)?;
    let time_b1 = fmodel.time_b1.reshape(&[1280, 1], &mut backend)?;
    let timef = fmodel.time_w1.matmul(&timef, &mut backend)?.add(&time_b1, &mut backend)?;
    let timef = silu(&timef, &mut backend)?;
    let timef = timef.reshape(&[1280, 1], &mut backend)?;
    let time_b2 = fmodel.time_b2.reshape(&[1280, 1], &mut backend)?;
    let timef = fmodel.time_w2.matmul(&timef, &mut backend)?.add(&time_b2, &mut backend)?;

    print!("\r 0/45 fconv_i0\x1b[K");    stdout().flush()?; let tensor_i0  = conv31       (&tensor,     &fmodel.fconv_i0.wc, &fmodel.fconv_i0.bc, &mut backend)?;
    print!("\r 1/45 frblock_i1\x1b[K");  stdout().flush()?; let tensor     = calc_frblock (&tensor_i0,  &timef, &fmodel.frblock_i1, &mut backend)?;
    print!("\r 2/45 fablock_i1\x1b[K");  stdout().flush()?; let tensor_i1  = calc_fablock (&tensor,     &context, &fmodel.fablock_i1, &mut backend)?;
    print!("\r 3/45 frblock_i2\x1b[K");  stdout().flush()?; let tensor     = calc_frblock (&tensor_i1,  &timef, &fmodel.frblock_i2, &mut backend)?;
    print!("\r 4/45 fablock_i2\x1b[K");  stdout().flush()?; let tensor_i2  = calc_fablock (&tensor,     &context, &fmodel.fablock_i2, &mut backend)?;
    print!("\r 5/45 fconv_i3\x1b[K");    stdout().flush()?; let tensor_i3  = conv32       (&tensor_i2,  &fmodel.fconv_i3.wc, &fmodel.fconv_i3.bc, &mut backend)?;
    print!("\r 6/45 frcblock_i4\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor_i3,  &timef, &fmodel.frcblock_i4, &mut backend)?;
    print!("\r 7/45 fablock_i4\x1b[K");  stdout().flush()?; let tensor_i4  = calc_fablock (&tensor,     &context, &fmodel.fablock_i4, &mut backend)?;
    print!("\r 8/45 frblock_i5\x1b[K");  stdout().flush()?; let tensor     = calc_frblock (&tensor_i4,  &timef, &fmodel.frblock_i5, &mut backend)?;
    print!("\r 9/45 fablock_i5\x1b[K");  stdout().flush()?; let tensor_i5  = calc_fablock (&tensor,     &context, &fmodel.fablock_i5, &mut backend)?;
    print!("\r10/45 fconv_i6\x1b[K");    stdout().flush()?; let tensor_i6  = conv32       (&tensor_i5,  &fmodel.fconv_i6.wc, &fmodel.fconv_i6.bc, &mut backend)?;
    print!("\r11/45 frcblock_i7\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor_i6,  &timef, &fmodel.frcblock_i7, &mut backend)?;
    print!("\r12/45 fablock_i7\x1b[K");  stdout().flush()?; let tensor_i7  = calc_fablock (&tensor,     &context, &fmodel.fablock_i7, &mut backend)?;
    print!("\r13/45 frblock_i8\x1b[K");  stdout().flush()?; let tensor     = calc_frblock (&tensor_i7,  &timef, &fmodel.frblock_i8, &mut backend)?;
    print!("\r14/45 fablock_i8\x1b[K");  stdout().flush()?; let tensor_i8  = calc_fablock (&tensor,     &context, &fmodel.fablock_i8, &mut backend)?;
    print!("\r15/45 fconv_i9\x1b[K");    stdout().flush()?; let tensor_i9  = conv32       (&tensor_i8,  &fmodel.fconv_i9.wc, &fmodel.fconv_i9.bc, &mut backend)?;
    print!("\r16/45 frblock_i10\x1b[K"); stdout().flush()?; let tensor_i10 = calc_frblock (&tensor_i9,  &timef, &fmodel.frblock_i10, &mut backend)?;
    print!("\r17/45 frblock_i11\x1b[K"); stdout().flush()?; let tensor_i11 = calc_frblock (&tensor_i10, &timef, &fmodel.frblock_i11, &mut backend)?;
    print!("\r18/45 frblock_m0\x1b[K");  stdout().flush()?; let tensor     = calc_frblock (&tensor_i11, &timef, &fmodel.frblock_m0, &mut backend)?;
    print!("\r19/45 fablock_m1\x1b[K");  stdout().flush()?; let tensor     = calc_fablock (&tensor,     &context, &fmodel.fablock_m1, &mut backend)?;
    print!("\r20/45 frblock_m2\x1b[K");  stdout().flush()?; let tensor     = calc_frblock (&tensor,     &timef, &fmodel.frblock_m2, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i11, &mut backend)?;
    print!("\r21/45 frcblock_o0\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o0, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i10, &mut backend)?;
    print!("\r22/45 frcblock_o1\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o1, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i9, &mut backend)?;
    print!("\r23/45 frcblock_o2\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o2, &mut backend)?;
    print!("\r24/45 fconv_o2\x1b[K");    stdout().flush()?; let tensor     = calc_upsample(&tensor,     &fmodel.fconv_o2, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i8, &mut backend)?;
    print!("\r25/45 frcblock_o3\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o3, &mut backend)?;
    print!("\r26/45 fablock_o3\x1b[K");  stdout().flush()?; let tensor     = calc_fablock (&tensor,     &context, &fmodel.fablock_o3, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i7, &mut backend)?;
    print!("\r27/45 frcblock_o4\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o4, &mut backend)?;
    print!("\r28/45 fablock_o4\x1b[K");  stdout().flush()?; let tensor     = calc_fablock (&tensor,     &context, &fmodel.fablock_o4, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i6, &mut backend)?;
    print!("\r29/45 frcblock_o5\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o5, &mut backend)?;
    print!("\r30/45 fablock_o5\x1b[K");  stdout().flush()?; let tensor     = calc_fablock (&tensor,     &context, &fmodel.fablock_o5, &mut backend)?;
    print!("\r31/45 fconv_o5\x1b[K");    stdout().flush()?; let tensor     = calc_upsample(&tensor,     &fmodel.fconv_o5, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i5, &mut backend)?;
    print!("\r32/45 frcblock_o6\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o6, &mut backend)?;
    print!("\r33/45 fablock_o6\x1b[K");  stdout().flush()?; let tensor     = calc_fablock (&tensor,     &context, &fmodel.fablock_o6, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i4, &mut backend)?;
    print!("\r34/45 frcblock_o7\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o7, &mut backend)?;
    print!("\r35/45 fablock_o7\x1b[K");  stdout().flush()?; let tensor     = calc_fablock (&tensor,     &context, &fmodel.fablock_o7, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i3, &mut backend)?;
    print!("\r36/45 frcblock_o8\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o8, &mut backend)?;
    print!("\r37/45 fablock_o8\x1b[K");  stdout().flush()?; let tensor     = calc_fablock (&tensor,     &context, &fmodel.fablock_o8, &mut backend)?;
    print!("\r38/45 fconv_o8\x1b[K");    stdout().flush()?; let tensor     = calc_upsample(&tensor,     &fmodel.fconv_o8, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i2, &mut backend)?;
    print!("\r39/45 frcblock_o9\x1b[K"); stdout().flush()?; let tensor     = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o9, &mut backend)?;
    print!("\r40/45 fablock_o9\x1b[K");  stdout().flush()?; let tensor     = calc_fablock (&tensor,     &context, &fmodel.fablock_o9, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i1, &mut backend)?;
    print!("\r41/45 frcblock_o10\x1b[K"); stdout().flush()?; let tensor    = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o10, &mut backend)?;
    print!("\r42/45 fablock_o10\x1b[K");  stdout().flush()?; let tensor    = calc_fablock (&tensor,     &context, &fmodel.fablock_o10, &mut backend)?;
    let tensor = cat(&tensor, &tensor_i0, &mut backend)?;
    print!("\r43/45 frcblock_o11\x1b[K"); stdout().flush()?; let tensor    = calc_frcblock(&tensor,     &timef, &fmodel.frcblock_o11, &mut backend)?;
    print!("\r44/45 fablock_o11\x1b[K");  stdout().flush()?; let tensor    = calc_fablock (&tensor,     &context, &fmodel.fablock_o11, &mut backend)?;
    print!("\r\x1b[K"); stdout().flush();

    let tensor = groupnorm(&tensor, &fmodel.g_final, &fmodel.t_final, 32, &mut backend)?;
    let tensor = silu(&tensor, &mut backend)?;
    let tensor = conv31(&tensor, &fmodel.wc_final, &fmodel.bc_final, &mut backend)?;

    let num_x = tensor.shape()[0];
    let num_y = tensor.shape()[1];
    let num_c = tensor.shape()[2];
    let num_n = tensor.shape()[3];
    let mut chunks = tensor.host_data()?.chunks(num_x * num_y * num_c);
    let tensor_positive = TypedTensor::<f32>::from_vec_col_major(
        vec![num_x, num_y, num_c, 1],
        chunks.next().unwrap().to_vec(),
    )?;
    let tensor_negative = TypedTensor::<f32>::from_vec_col_major(
        vec![num_x, num_y, num_c, 1],
        chunks.next().unwrap().to_vec(),
    )?;
    let config_scale = TypedTensor::<f32>::from_vec_col_major(vec![], vec![8.0])?;
    let tensor = tensor_positive.sub(&tensor_negative, &mut backend)?.mul(&config_scale, &mut backend)?.add(&tensor_negative, &mut backend)?;

    // 1 .- range(√0.00085f0, √0.0120f0, 1000) .^ 2
    let alphas: Vec<f32> = (0..1000).map(
        |a| 1.0f32 - (0.00085f32.sqrt() + (0.0120f32.sqrt() - 0.00085f32.sqrt()) * ((a as f32) / 999.0f32)).powi(2)
    ).collect();

    let cumprod_alphas: Vec<f32> = {
        let mut v = Vec::<f32>::with_capacity(1000);
        let mut prod = 1.0f32;
        for a in alphas {
            prod *= a;
            v.push(prod);
        }
        v
    };

    let curr_alpha_bar = cumprod_alphas[curr_time as usize];
    let prev_alpha_bar = if prev_time >= 0 { cumprod_alphas[prev_time as usize] } else { 1.0f32 };
    let alpha_t = curr_alpha_bar / prev_alpha_bar;

    // References: "Denoising Diffusion Probabilistic Models"
    // https://arxiv.org/pdf/2006.11239

    // xₜ -> tensor_orig
    // ϵ -> tensor

    let c0 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![(1.0f32 - curr_alpha_bar).sqrt()])?;
    let c1 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![curr_alpha_bar.sqrt()])?;
    let y0 = tensor.mul(&c0, &mut backend)?;
    let x0 = tensor_orig.sub(&y0, &mut backend)?.div(&c1, &mut backend)?;

    let c2 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![prev_alpha_bar.sqrt() * (1.0f32 - alpha_t) / (1.0f32 - curr_alpha_bar)])?;
    let c3 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![alpha_t.sqrt() * (1.0f32 - prev_alpha_bar) / (1.0f32 - curr_alpha_bar)])?;
    let y2 = x0.mul(&c2, &mut backend)?;
    let y3 = tensor_orig.mul(&c3, &mut backend)?;
    let mu_t = y2.add(&y3, &mut backend)?;

    let tensor = if curr_time > 0 {
        let noise = randn(vec![64, 64, 4, 1], rng)?;
        let c4 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![((1.0f32 - prev_alpha_bar) / (1.0f32 - curr_alpha_bar) * (1.0 - alpha_t)).sqrt()])?;
        noise.mul(&c4, &mut backend)?.add(&mu_t, &mut backend)?
    } else {
        mu_t
    };

    Ok(tensor)
    // U-net done!
}
