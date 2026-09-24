// The output of the U-net is latent-encoded.
// The decoder decodes the latent-encoded tensor into RGB images
// so that you can see the result!

use std::error::Error;
use std::io::{Write, stdout};

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

use crate::model::{Dablock, Dmodel, Drblock, Drcblock};
use crate::util::{conv11, conv31, groupnorm, self_attention, silu, show, upsample};

pub fn calc_drblock(tensor: &TypedTensor<f32>, drblock: &Drblock, backend: &mut CpuBackend) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let tmp = groupnorm(tensor, &drblock.g1, &drblock.t1, 32, backend)?;
    let tmp = silu(&tmp, backend)?;
    let tmp = conv31(&tmp, &drblock.wc1, &drblock.bc1, backend)?;
    let tmp = groupnorm(&tmp, &drblock.g2, &drblock.t2, 32, backend)?;
    let tmp = silu(&tmp, backend)?;
    let tmp = conv31(&tmp, &drblock.wc2, &drblock.bc2, backend)?;
    let tensor = tensor.add(&tmp, backend)?;
    Ok(tensor)
}

pub fn calc_drcblock(tensor: &TypedTensor<f32>, drcblock: &Drcblock, backend: &mut CpuBackend) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let tmp = groupnorm(tensor, &drcblock.g1, &drcblock.t1, 32, backend)?;
    let tmp = silu(&tmp, backend)?;
    let tmp = conv31(&tmp, &drcblock.wc1, &drcblock.bc1, backend)?;
    let tmp = groupnorm(&tmp, &drcblock.g2, &drcblock.t2, 32, backend)?;
    let tmp = silu(&tmp, backend)?;
    let tmp = conv31(&tmp, &drcblock.wc2, &drcblock.bc2, backend)?;
    let tensor = conv11(&tensor, &drcblock.wc3, &drcblock.bc3, backend)?;
    let tensor = tensor.add(&tmp, backend)?;
    Ok(tensor)
}

pub fn calc_dablock(tensor: &TypedTensor<f32>, dablock: &Dablock, backend: &mut CpuBackend) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let tmp = groupnorm(tensor, &dablock.g, &dablock.t, 32, backend)?;
    let tmp = self_attention(&tmp, &dablock.w1, &dablock.b1, &dablock.w2, &dablock.b2, 1, backend)?;
    let tensor = tensor.add(&tmp, backend)?;
    Ok(tensor)
}

pub fn decode(tensor: &TypedTensor<f32>, dmodel: Dmodel) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let c = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.18215])?;
    let tensor = tensor.div(&c, &mut backend)?;

    print!("\rd  0/26 dconv_pq\x1b[K");     stdout().flush(); let tensor = conv11(&tensor, &dmodel.dconv_pq.wc, &dmodel.dconv_pq.bc, &mut backend)?;
    print!("\rd  1/26 dconv_in\x1b[K");     stdout().flush(); let tensor = conv31(&tensor, &dmodel.dconv_in.wc, &dmodel.dconv_in.bc, &mut backend)?;
    print!("\rd  2/26 drblock_mid1\x1b[K"); stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_mid1, &mut backend)?;
    print!("\rd  3/26 dablock\x1b[K");      stdout().flush(); let tensor = calc_dablock(&tensor, &dmodel.dablock, &mut backend)?;
    print!("\rd  4/26 drblock_mid2\x1b[K"); stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_mid2, &mut backend)?;
    print!("\rd  5/26 drblock_30\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_30, &mut backend)?;
    print!("\rd  6/26 drblock_31\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_31, &mut backend)?;
    print!("\rd  7/26 drblock_32\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_32, &mut backend)?;
    print!("\rd  8/26 upsample\x1b[K");     stdout().flush(); let tensor = upsample(&tensor)?;
    print!("\rd  9/26 dconv_3\x1b[K");      stdout().flush(); let tensor = conv31(&tensor, &dmodel.dconv_3.wc, &dmodel.dconv_3.bc, &mut backend)?;
    print!("\rd 10/26 drblock_20\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_20, &mut backend)?;
    print!("\rd 11/26 drblock_21\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_21, &mut backend)?;
    print!("\rd 12/26 drblock_22\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_22, &mut backend)?;
    print!("\rd 13/26 upsample\x1b[K");     stdout().flush(); let tensor = upsample(&tensor)?;
    print!("\rd 14/26 dconv_2\x1b[K");      stdout().flush(); let tensor = conv31(&tensor, &dmodel.dconv_2.wc, &dmodel.dconv_2.bc, &mut backend)?;
    print!("\rd 15/26 drcblock_10\x1b[K");  stdout().flush(); let tensor = calc_drcblock(&tensor, &dmodel.drcblock_10, &mut backend)?;
    print!("\rd 16/26 drblock_11\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_11, &mut backend)?;
    print!("\rd 17/26 drblock_12\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_12, &mut backend)?;
    print!("\rd 18/26 upsample\x1b[K");     stdout().flush(); let tensor = upsample(&tensor)?;
    print!("\rd 19/26 dconv_1\x1b[K");      stdout().flush(); let tensor = conv31(&tensor, &dmodel.dconv_1.wc, &dmodel.dconv_1.bc, &mut backend)?;
    print!("\rd 20/26 drcblock_00\x1b[K");  stdout().flush(); let tensor = calc_drcblock(&tensor, &dmodel.drcblock_00, &mut backend)?;
    print!("\rd 21/26 drblock_01\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_01, &mut backend)?;
    print!("\rd 22/26 drblock_02\x1b[K");   stdout().flush(); let tensor = calc_drblock(&tensor, &dmodel.drblock_02, &mut backend)?;
    print!("\rd 23/26 groupnorm\x1b[K");    stdout().flush(); let tensor = groupnorm(&tensor, &dmodel.g, &dmodel.t, 32, &mut backend)?;
    print!("\rd 24/26 silu\x1b[K");         stdout().flush(); let tensor = silu(&tensor, &mut backend)?;
    print!("\rd 25/26 dconv_out\x1b[K");    stdout().flush(); let tensor = conv31(&tensor, &dmodel.dconv_out.wc, &dmodel.dconv_out.bc, &mut backend)?;
    print!("\r");                           stdout().flush();

    Ok(tensor)
}
