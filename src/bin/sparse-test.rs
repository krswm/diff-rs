// I'll use sparse tensor for 2D matrix convolution.
// How do I use sparse tensor with tenferro?
// Without sparse tensor I have to create tensor with L^4
// (L = length of a side of an image) elements although most of them
// are zero.

// Official documentation:
// https://tensor4all.org/tenferro-rs/tutorials/sparse-extension.html

use std::error::Error;

// use tenferro_ext_sparse::{SparseCooTensor, sparse_matmul_eager};
// ??? What crate do I have to add to Cargo.toml?
// ---
// Reading the sparse's Cargo.toml, I find it has `publish = false`
// https://github.com/tensor4all/tenferro-rs/blob/main/ext/sparse/Cargo.toml
// ---
// I learned there's `git` option in Cargo.toml!

// use tenferro_runtime::Tensor;
// use tenferro_tensor::types::Tensor;
use rand::SeedableRng;
use rand::rngs::ChaCha20Rng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

fn main() -> Result<(), Box<dyn Error>> {
    /*
    // 2.0 1.0
    // 3.0 0.0

    // The indices of the non-zero values are: (0, 0), (0, 1), (1, 0)
    let coords = Tensor::from_vec_col_major(vec![2, 3], vec![0i64, 0, 0, 1, 1, 0])?;

    // Those non-zero values are: 2.0, 1.0, 3.0
    let values = Tensor::from_vec_col_major(vec![3], vec![2.0, 1.0, 3.0])?;

    // The shape of the tensor is: (2, 2)
    let sparse = SparseCooTensor::from_parts(vec![2, 2], coords, values)?;

    println!("{sparse:?}");

    // let identity = Tensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;

    // let backend = CpuBackend::new();

    // let result = sparse.matmul(&identity, &mut backend)?;  // No?

    // let result = [&sparse, &identity].einsum("ij,jk->ik", &mut backend)?;  // No?

    let identity = SparseCooTensor::from_parts(
        vec![2, 2],
        Tensor::from_vec_col_major(vec![2, 2], vec![0i64,0, 1,1])?,
        Tensor::from_vec_col_major(vec![2], vec![1.0, 1.0])?,
    )?;

    // let result = [&sparse, &identity].einsum("ij,jk->ik", &mut backend)?;  // No?

    // let result = sparse.mul(identity, &mut backend)?;  // No?

    // let result = sparse_matmul_eager(&sparse, &identity)?;  // OK!

    // 2 1  .  2 1  =  7 2
    // 3 0     3 0     6 3

    let result = sparse_matmul_eager(&sparse, &sparse)?;

    // println!("{result:?}");

    // println!("{}", result.get(&[0, 0])?);  // No?

    */

    // Let me simulate matmul of huge tensors.

    let L = 64;
    let C = 320;
    let N = L * L;

    let distr = Normal::new(0.0, 1.0)?;
    let mut rng = ChaCha20Rng::seed_from_u64(2269);

    // let colmaj: Vec<f32> = distr.sample_iter(&mut rng).take(N).collect();
    let colmaj: Vec<f32> = (0..(N * C)).into_iter().map(|a| a as f32).collect();
    let a = TypedTensor::<f32>::from_vec_col_major(vec![N, C], colmaj)?;

    // let colmaj: Vec<f32> = distr.sample_iter(&mut rng).take(N * N).collect();
    let colmaj: Vec<f32> = (0..(N * N * C)).into_iter().map(|a| a as f32).collect();
    let b = TypedTensor::<f32>::from_vec_col_major(vec![N, N, C], colmaj)?;

    println!("start");
    let performance_timer = Instant::now();
    let mut backend = CpuBackend::new();
    let result_ = [&b, &a].einsum("ijc,jc->ic", &mut backend)?;
    let performance_time = performance_timer.elapsed().as_secs_f64();
    println!("{performance_time} s");

    // 16.448 s for L = 64, C = 320 in a quick test on a hardware I have access to.

    // Maybe I have to look for creating an extension?
    // https://tensor4all.org/tenferro-rs/guides/custom-operations.html

    Ok(())
}
