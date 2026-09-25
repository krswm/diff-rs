// My U-net implementation with tenferro!

use std::error::Error;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

use crate::model::Fmodel;
use crate::util::show;

pub fn forward(
    tensor: &TypedTensor<f32>,
    context: &TypedTensor<f32>,
    curr_time: i32,
    prev_time: i32,
    fmodel: &Fmodel,
) -> Result<(), Box<dyn Error>> {
    let mut backend = CpuBackend::new();
    
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

    show(&timef)?;
    
    Ok(())
}
