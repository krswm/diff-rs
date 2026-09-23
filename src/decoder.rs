// The output of the U-net is latent-encoded.
// The decoder decodes the latent-encoded tensor into RGB images
// so that you can see the result!

use std::error::Error;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

use crate::model::Dmodel;
use crate::util::show;

pub fn decode(tensor: &TypedTensor<f32>, dmodel: Dmodel) -> Result<(), Box<dyn Error>> {
    let mut backend = CpuBackend::new();

    show(tensor)?;
    println!();

    let c = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.18215])?;
    let tensor = tensor.div(&c, &mut backend)?;
    show(&tensor)?;
    println!();

    Ok(())
}
