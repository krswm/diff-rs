// Machine Learning Utilities

// I'm going to build smaller parts first.
// What do I need?
// - show
//   Prints the edges of a tensor, for example, if A is 30x40 matrix show A[0, 0], A[29, 0], A[0, 39], and A[29, 39].
//   so that I can compare the computation with reference implementation.
// - randn
//   Generates a tensor whose elements are random numbers sampled from the normal distribution.
// - Nonlinear activation functions
//   - silu
//   - gelu
// - Normalizations
//   - layernorm
//   - groupnorm
// - Image processing
//   - conv
//     - kernel: 1x1, stride: 1
//     - kernel: 3x3, stride: 1
//     - kernel: 3x3, stride: 2
//   - upsample
// - Attention
//   - self_attention
//   - cross_attention

use std::error::Error;

use rand::SeedableRng;
use rand::rngs::ChaCha20Rng;
use rand_distr::{Distribution, Normal};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

#[allow(dead_code)]
fn show(tensor: &TypedTensor<f32>) -> Result<(), Box<dyn Error>> {
    // TODO: Not elegant.
    if tensor.rank() == 0 {
        println!("[]: {:+15.6e}", tensor.get(&[])?);
    } else if tensor.rank() == 1 {
        let shape = tensor.shape();
        for i0 in [0, shape[0] - 1] {
            println!("[{:6}]: {:<+.6e}", i0, tensor.get(&[i0])?);
        }
    } else if tensor.rank() == 2 {
        let shape = tensor.shape();
        for (i0, i1) in [
            (0, 0),
            (shape[0] - 1, 0),
            (0, shape[1] - 1),
            (shape[0] - 1, shape[1] - 1),
        ] {
            println!("[{:6}, {:6}]: {:<+.6e}", i0, i1, tensor.get(&[i0, i1])?);
        }
    } else if tensor.rank() == 3 {
        let shape = tensor.shape();
        for (i0, i1, i2) in [
            (0, 0, 0),
            (shape[0] - 1, 0, 0),
            (0, shape[1] - 1, 0),
            (shape[0] - 1, shape[1] - 1, 0),
            (0, 0, shape[2] - 1),
            (shape[0] - 1, 0, shape[2] - 1),
            (0, shape[1] - 1, shape[2] - 1),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1),
        ] {
            println!(
                "[{:6}, {:6}, {:6}]: {:<+.6e}",
                i0,
                i1,
                i2,
                tensor.get(&[i0, i1, i2])?
            );
        }
    } else if tensor.rank() == 4 {
        let shape = tensor.shape();
        for (i0, i1, i2, i3) in [
            (0, 0, 0, 0),
            (shape[0] - 1, 0, 0, 0),
            (0, shape[1] - 1, 0, 0),
            (shape[0] - 1, shape[1] - 1, 0, 0),
            (0, 0, shape[2] - 1, 0),
            (shape[0] - 1, 0, shape[2] - 1, 0),
            (0, shape[1] - 1, shape[2] - 1, 0),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, 0),
            (0, 0, 0, shape[3] - 1),
            (shape[0] - 1, 0, 0, shape[3] - 1),
            (0, shape[1] - 1, 0, shape[3] - 1),
            (shape[0] - 1, shape[1] - 1, 0, shape[3] - 1),
            (0, 0, shape[2] - 1, shape[3] - 1),
            (shape[0] - 1, 0, shape[2] - 1, shape[3] - 1),
            (0, shape[1] - 1, shape[2] - 1, shape[3] - 1),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1),
        ] {
            println!(
                "[{:6}, {:6}, {:6}, {:6}]: {:<+.6e}",
                i0,
                i1,
                i2,
                i3,
                tensor.get(&[i0, i1, i2, i3])?
            );
        }
    } else if tensor.rank() == 5 {
        let shape = tensor.shape();
        for (i0, i1, i2, i3, i4) in [
            (0, 0, 0, 0, 0),
            (shape[0] - 1, 0, 0, 0, 0),
            (0, shape[1] - 1, 0, 0, 0),
            (shape[0] - 1, shape[1] - 1, 0, 0, 0),
            (0, 0, shape[2] - 1, 0, 0),
            (shape[0] - 1, 0, shape[2] - 1, 0, 0),
            (0, shape[1] - 1, shape[2] - 1, 0, 0),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0),
            (0, 0, 0, shape[3] - 1, 0),
            (shape[0] - 1, 0, 0, shape[3] - 1, 0),
            (0, shape[1] - 1, 0, shape[3] - 1, 0),
            (shape[0] - 1, shape[1] - 1, 0, shape[3] - 1, 0),
            (0, 0, shape[2] - 1, shape[3] - 1, 0),
            (shape[0] - 1, 0, shape[2] - 1, shape[3] - 1, 0),
            (0, shape[1] - 1, shape[2] - 1, shape[3] - 1, 0),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1, 0),
            (0, 0, 0, 0, shape[4] - 1),
            (shape[0] - 1, 0, 0, 0, shape[4] - 1),
            (0, shape[1] - 1, 0, 0, shape[4] - 1),
            (shape[0] - 1, shape[1] - 1, 0, 0, shape[4] - 1),
            (0, 0, shape[2] - 1, 0, shape[4] - 1),
            (shape[0] - 1, 0, shape[2] - 1, 0, shape[4] - 1),
            (0, shape[1] - 1, shape[2] - 1, 0, shape[4] - 1),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, shape[4] - 1),
            (0, 0, 0, shape[3] - 1, shape[4] - 1),
            (shape[0] - 1, 0, 0, shape[3] - 1, shape[4] - 1),
            (0, shape[1] - 1, 0, shape[3] - 1, shape[4] - 1),
            (shape[0] - 1, shape[1] - 1, 0, shape[3] - 1, shape[4] - 1),
            (0, 0, shape[2] - 1, shape[3] - 1, shape[4] - 1),
            (shape[0] - 1, 0, shape[2] - 1, shape[3] - 1, shape[4] - 1),
            (0, shape[1] - 1, shape[2] - 1, shape[3] - 1, shape[4] - 1),
            (
                shape[0] - 1,
                shape[1] - 1,
                shape[2] - 1,
                shape[3] - 1,
                shape[4] - 1,
            ),
        ] {
            println!(
                "[{:6}, {:6}, {:6}, {:6}, {:6}]: {:<+.6e}",
                i0,
                i1,
                i2,
                i3,
                i4,
                tensor.get(&[i0, i1, i2, i3, i4])?
            );
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn randn(shape: Vec<usize>, rng: &mut ChaCha20Rng) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // Generates a tensor whose elements are random numbers sampled from the normal distribution.
    let n_elements = shape.iter().copied().reduce(|a, b| a * b).unwrap();
    let distr = Normal::new(0.0, 1.0)?;
    let colmaj = distr.sample_iter(rng).take(n_elements).collect();
    let tensor = TypedTensor::<f32>::from_vec_col_major(shape, colmaj)?;
    Ok(tensor)
}

#[allow(dead_code)]
fn silu(
    tensor: &TypedTensor<f32>,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let one = TypedTensor::<f32>::from_vec_col_major(vec![], vec![1.0])?;
    let denominator = tensor.neg(backend)?.exp(backend)?.add(&one, backend)?;
    let tensor = tensor.div(&denominator, backend)?;
    Ok(tensor)
}

#[allow(dead_code)]
fn gelu(
    tensor: &TypedTensor<f32>,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // This formula is based on the paper that introduced GELU.
    // https://arxiv.org/abs/1606.08415
    // x = (tanh.((x .^ 3 * 0.044715f0 + x) * √(2.0f0 / π)) .+ 1.0f0) .* x * 0.5f0
    let c0 = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.044715f32])?;
    let c1 = {
        let value = (2.0f32 / std::f32::consts::PI).sqrt();
        TypedTensor::<f32>::from_vec_col_major(vec![], vec![value])?
    };
    let one = TypedTensor::<f32>::from_vec_col_major(vec![], vec![1.0f32])?;
    let half = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.5f32])?;
    let tensor = tensor
        .mul(tensor, backend)?
        .mul(tensor, backend)?
        .mul(&c0, backend)?
        .add(tensor, backend)?
        .mul(&c1, backend)?
        .tanh(backend)?
        .add(&one, backend)?
        .mul(tensor, backend)?
        .mul(&half, backend)?;
    Ok(tensor)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = ChaCha20Rng::seed_from_u64(2269);
    let tensor = randn(vec![2, 2], &mut rng)?;
    show(&tensor)?;
    Ok(())
}
