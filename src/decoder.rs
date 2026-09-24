// The output of the U-net is latent-encoded.
// The decoder decodes the latent-encoded tensor into RGB images
// so that you can see the result!

use std::error::Error;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

use crate::model::{Dablock, Dmodel, Drblock};
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

pub fn calc_dablock(tensor: &TypedTensor<f32>, dablock: &Dablock, backend: &mut CpuBackend) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let tmp = groupnorm(tensor, &dablock.g, &dablock.t, 32, backend)?;
    let tmp = self_attention(&tmp, &dablock.w1, &dablock.b1, &dablock.w2, &dablock.b2, 1, backend)?;
    let tensor = tensor.add(&tmp, backend)?;
    Ok(tensor)
}

pub fn decode(tensor: &TypedTensor<f32>, dmodel: Dmodel) -> Result<(), Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    let c = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.18215])?;
    let tensor = tensor.div(&c, &mut backend)?;

    let tensor = conv11(&tensor, &dmodel.dconv_pq.wc, &dmodel.dconv_pq.bc, &mut backend)?;
    let tensor = conv31(&tensor, &dmodel.dconv_in.wc, &dmodel.dconv_in.bc, &mut backend)?;
    let tensor = calc_drblock(&tensor, &dmodel.drblock_mid1, &mut backend)?;
    let tensor = calc_dablock(&tensor, &dmodel.dablock, &mut backend)?;
    let tensor = calc_drblock(&tensor, &dmodel.drblock_mid2, &mut backend)?;
    let tensor = calc_drblock(&tensor, &dmodel.drblock_30, &mut backend)?;
    let tensor = calc_drblock(&tensor, &dmodel.drblock_31, &mut backend)?;
    let tensor = calc_drblock(&tensor, &dmodel.drblock_32, &mut backend)?;
    let tensor = upsample(&tensor)?;
    let tensor = conv31(&tensor, &dmodel.dconv_3.wc, &dmodel.dconv_3.bc, &mut backend)?;

    show(&tensor)?;

    Ok(())
}
