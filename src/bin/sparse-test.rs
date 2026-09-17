// I'll use sparse tensor for 2D matrix convolution.
// How do I use sparse tensor with tenferro?
// Without sparse tensor I have to create tensor with L^4
// (L = length of a side of an image) elements although most of them
// are zero.

// Official documentation:
// https://tensor4all.org/tenferro-rs/tutorials/sparse-extension.html

use std::error::Error;

use tenferro_ext_sparse::SparseCooTensor;
// ??? What crate do I have to add to Cargo.toml?
// ---
// Reading the sparse's Cargo.toml, I find it has `publish = false`
// https://github.com/tensor4all/tenferro-rs/blob/main/ext/sparse/Cargo.toml
// ---
// I learned there's `git` option in Cargo.toml!

// use tenferro_runtime::Tensor;
use tenferro_tensor::types::Tensor;
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn Error>> {
    // 2.0 1.0
    // 3.0 0.0
    
    // The indices of the non-zero values are: (0, 0), (0, 1), (1, 0)
    let coords = Tensor::from_vec_col_major(vec![2, 3], vec![0i64, 0, 0, 1, 1, 0])?;

    // Those non-zero values are: 2.0, 1.0, 3.0
    let values = Tensor::from_vec_col_major(vec![3], vec![2.0, 1.0, 3.0])?;

    // The shape of the tensor is: (2, 2)
    let sparse = SparseCooTensor::from_parts(vec![2, 2], coords, values)?;
    
    println!("{sparse:?}");

    let identity = Tensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;

    let backend = CpuBackend::new();

    // let result = sparse.matmul(&identity, &mut backend)?;  // No?

    println!("{result:?}");

    Ok(())
}
