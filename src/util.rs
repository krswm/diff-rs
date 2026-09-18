// Machine Learning Utilities

// I'm going to build smaller parts first.
// What do I need?
// - show
//   Prints the edges of a tensor, for example, if A is 30x40 matrix show A[0, 0], A[29, 0], A[0, 39], and A[29, 39].
//   so that I can compare the computation with reference implementation.
// - randn
//   Generates a tensor whose elements are random numbers sampled from the normal distribution.
// - softmax
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
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

pub fn show(tensor: &TypedTensor<f32>) -> Result<(), Box<dyn Error>> {
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
    } else if tensor.rank() == 6 {
        let shape = tensor.shape();
        for (i0, i1, i2, i3, i4, i5) in [
            (0, 0, 0, 0, 0, 0),
            (shape[0] - 1, 0, 0, 0, 0, 0),
            (0, shape[1] - 1, 0, 0, 0, 0),
            (shape[0] - 1, shape[1] - 1, 0, 0, 0, 0),
            (0, 0, shape[2] - 1, 0, 0, 0),
            (shape[0] - 1, 0, shape[2] - 1, 0, 0, 0),
            (0, shape[1] - 1, shape[2] - 1, 0, 0, 0),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0),
            (0, 0, 0, shape[3] - 1, 0, 0),
            (shape[0] - 1, 0, 0, shape[3] - 1, 0, 0),
            (0, shape[1] - 1, 0, shape[3] - 1, 0, 0),
            (shape[0] - 1, shape[1] - 1, 0, shape[3] - 1, 0, 0),
            (0, 0, shape[2] - 1, shape[3] - 1, 0, 0),
            (shape[0] - 1, 0, shape[2] - 1, shape[3] - 1, 0, 0),
            (0, shape[1] - 1, shape[2] - 1, shape[3] - 1, 0, 0),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1, 0, 0),
            (0, 0, 0, 0, shape[4] - 1, 0),
            (shape[0] - 1, 0, 0, 0, shape[4] - 1, 0),
            (0, shape[1] - 1, 0, 0, shape[4] - 1, 0),
            (shape[0] - 1, shape[1] - 1, 0, 0, shape[4] - 1, 0),
            (0, 0, shape[2] - 1, 0, shape[4] - 1, 0),
            (shape[0] - 1, 0, shape[2] - 1, 0, shape[4] - 1, 0),
            (0, shape[1] - 1, shape[2] - 1, 0, shape[4] - 1, 0),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, shape[4] - 1, 0),
            (0, 0, 0, shape[3] - 1, shape[4] - 1, 0),
            (shape[0] - 1, 0, 0, shape[3] - 1, shape[4] - 1, 0),
            (0, shape[1] - 1, 0, shape[3] - 1, shape[4] - 1, 0),
            (shape[0] - 1, shape[1] - 1, 0, shape[3] - 1, shape[4] - 1, 0),
            (0, 0, shape[2] - 1, shape[3] - 1, shape[4] - 1, 0),
            (shape[0] - 1, 0, shape[2] - 1, shape[3] - 1, shape[4] - 1, 0),
            (0, shape[1] - 1, shape[2] - 1, shape[3] - 1, shape[4] - 1, 0),
            (
                shape[0] - 1,
                shape[1] - 1,
                shape[2] - 1,
                shape[3] - 1,
                shape[4] - 1,
                0,
            ),
            (0, 0, 0, 0, 0, shape[5] - 1),
            (shape[0] - 1, 0, 0, 0, 0, shape[5] - 1),
            (0, shape[1] - 1, 0, 0, 0, shape[5] - 1),
            (shape[0] - 1, shape[1] - 1, 0, 0, 0, shape[5] - 1),
            (0, 0, shape[2] - 1, 0, 0, shape[5] - 1),
            (shape[0] - 1, 0, shape[2] - 1, 0, 0, shape[5] - 1),
            (0, shape[1] - 1, shape[2] - 1, 0, 0, shape[5] - 1),
            (shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, shape[5] - 1),
            (0, 0, 0, shape[3] - 1, 0, shape[5] - 1),
            (shape[0] - 1, 0, 0, shape[3] - 1, 0, shape[5] - 1),
            (0, shape[1] - 1, 0, shape[3] - 1, 0, shape[5] - 1),
            (shape[0] - 1, shape[1] - 1, 0, shape[3] - 1, 0, shape[5] - 1),
            (0, 0, shape[2] - 1, shape[3] - 1, 0, shape[5] - 1),
            (shape[0] - 1, 0, shape[2] - 1, shape[3] - 1, 0, shape[5] - 1),
            (0, shape[1] - 1, shape[2] - 1, shape[3] - 1, 0, shape[5] - 1),
            (
                shape[0] - 1,
                shape[1] - 1,
                shape[2] - 1,
                shape[3] - 1,
                0,
                shape[5] - 1,
            ),
            (0, 0, 0, 0, shape[4] - 1, shape[5] - 1),
            (shape[0] - 1, 0, 0, 0, shape[4] - 1, shape[5] - 1),
            (0, shape[1] - 1, 0, 0, shape[4] - 1, shape[5] - 1),
            (shape[0] - 1, shape[1] - 1, 0, 0, shape[4] - 1, shape[5] - 1),
            (0, 0, shape[2] - 1, 0, shape[4] - 1, shape[5] - 1),
            (shape[0] - 1, 0, shape[2] - 1, 0, shape[4] - 1, shape[5] - 1),
            (0, shape[1] - 1, shape[2] - 1, 0, shape[4] - 1, shape[5] - 1),
            (
                shape[0] - 1,
                shape[1] - 1,
                shape[2] - 1,
                0,
                shape[4] - 1,
                shape[5] - 1,
            ),
            (0, 0, 0, shape[3] - 1, shape[4] - 1, shape[5] - 1),
            (shape[0] - 1, 0, 0, shape[3] - 1, shape[4] - 1, shape[5] - 1),
            (0, shape[1] - 1, 0, shape[3] - 1, shape[4] - 1, shape[5] - 1),
            (
                shape[0] - 1,
                shape[1] - 1,
                0,
                shape[3] - 1,
                shape[4] - 1,
                shape[5] - 1,
            ),
            (0, 0, shape[2] - 1, shape[3] - 1, shape[4] - 1, shape[5] - 1),
            (
                shape[0] - 1,
                0,
                shape[2] - 1,
                shape[3] - 1,
                shape[4] - 1,
                shape[5] - 1,
            ),
            (
                0,
                shape[1] - 1,
                shape[2] - 1,
                shape[3] - 1,
                shape[4] - 1,
                shape[5] - 1,
            ),
            (
                shape[0] - 1,
                shape[1] - 1,
                shape[2] - 1,
                shape[3] - 1,
                shape[4] - 1,
                shape[5] - 1,
            ),
        ] {
            println!(
                "[{:6}, {:6}, {:6}, {:6}, {:6}, {:6}]: {:<+.6e}",
                i0,
                i1,
                i2,
                i3,
                i4,
                i5,
                tensor.get(&[i0, i1, i2, i3, i4, i5])?
            );
        }
    }
    Ok(())
}

pub fn cshow(tensor: &TypedTensor<f32>) -> Result<(), Box<dyn Error>> {
    // For debugging `conv`.

    let num_rows = tensor.shape()[0];
    let num_cols = tensor.shape()[1];

    println!(
        "┏{:━^15}┯{:━^15}┯{:━^15}┯{:━^15}┯{:━^15}┯{:━^15}┯{:━^15}┓ ╮",
        "", "", "", "", "", "", ""
    );
    println!(
        "┃ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} │ {:^13} │ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} ┃ │",
        tensor.get(&[0, 0, 0, 0]).unwrap(),
        tensor.get(&[0, 1, 0, 0]).unwrap(),
        tensor.get(&[0, 2, 0, 0]).unwrap(),
        "⋯",
        tensor.get(&[0, num_cols - 3, 0, 0]).unwrap(),
        tensor.get(&[0, num_cols - 2, 0, 0]).unwrap(),
        tensor.get(&[0, num_cols - 1, 0, 0]).unwrap(),
    );
    println!(
        "┠{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┨ │",
        "", "", "", "", "", "", ""
    );
    println!(
        "┃ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} │ {:^13} │ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} ┃ │",
        tensor.get(&[1, 0, 0, 0]).unwrap(),
        tensor.get(&[1, 1, 0, 0]).unwrap(),
        tensor.get(&[1, 2, 0, 0]).unwrap(),
        "⋯",
        tensor.get(&[1, num_cols - 3, 0, 0]).unwrap(),
        tensor.get(&[1, num_cols - 2, 0, 0]).unwrap(),
        tensor.get(&[1, num_cols - 1, 0, 0]).unwrap(),
    );
    println!(
        "┠{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┨ │",
        "", "", "", "", "", "", ""
    );
    println!(
        "┃ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} │ {:^13} │ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} ┃ │",
        tensor.get(&[2, 0, 0, 0]).unwrap(),
        tensor.get(&[2, 1, 0, 0]).unwrap(),
        tensor.get(&[2, 2, 0, 0]).unwrap(),
        "⋯",
        tensor.get(&[2, num_cols - 3, 0, 0]).unwrap(),
        tensor.get(&[2, num_cols - 2, 0, 0]).unwrap(),
        tensor.get(&[2, num_cols - 1, 0, 0]).unwrap(),
    );
    println!(
        "┠{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┨ │",
        "", "", "", "", "", "", ""
    );
    println!(
        "┃ {:^13} │ {:^13} │ {:^13} │ {:13} │ {:^13} │ {:^13} │ {:^13} ┃ {num_rows}",
        "⋮", "⋮", "⋮", "", "⋮", "⋮", "⋮"
    );
    println!(
        "┠{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┨ │",
        "", "", "", "", "", "", ""
    );
    println!(
        "┃ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} │ {:^13} │ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} ┃ │",
        tensor.get(&[num_rows - 3, 0, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 3, 1, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 3, 2, 0, 0]).unwrap(),
        "⋯",
        tensor.get(&[num_rows - 3, num_cols - 3, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 3, num_cols - 2, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 3, num_cols - 1, 0, 0]).unwrap(),
    );
    println!(
        "┠{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┨ │",
        "", "", "", "", "", "", ""
    );
    println!(
        "┃ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} │ {:^13} │ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} ┃ │",
        tensor.get(&[num_rows - 2, 0, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 2, 1, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 2, 2, 0, 0]).unwrap(),
        "⋯",
        tensor.get(&[num_rows - 2, num_cols - 3, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 2, num_cols - 2, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 2, num_cols - 1, 0, 0]).unwrap(),
    );
    println!(
        "┠{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┼{:─^15}┨ │",
        "", "", "", "", "", "", ""
    );
    println!(
        "┃ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} │ {:^13} │ {:<+13.6e} │ {:<+13.6e} │ {:<+13.6e} ┃ │",
        tensor.get(&[num_rows - 1, 0, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 1, 1, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 1, 2, 0, 0]).unwrap(),
        "⋯",
        tensor.get(&[num_rows - 1, num_cols - 3, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 1, num_cols - 2, 0, 0]).unwrap(),
        tensor.get(&[num_rows - 1, num_cols - 1, 0, 0]).unwrap(),
    );
    println!(
        "┗{:━^15}┷{:━^15}┷{:━^15}┷{:━^15}┷{:━^15}┷{:━^15}┷{:━^15}┛ ╯",
        "", "", "", "", "", "", ""
    );
    println!("╰{num_cols:─^111}╯");

    Ok(())
}

pub fn randn(shape: Vec<usize>, rng: &mut ChaCha20Rng) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // Generates a tensor whose elements are random numbers sampled from the normal distribution.
    let n_elements = shape.iter().copied().reduce(|a, b| a * b).unwrap();
    let distr = Normal::new(0.0, 1.0)?;
    let colmaj = distr.sample_iter(rng).take(n_elements).collect();
    let tensor = TypedTensor::<f32>::from_vec_col_major(shape, colmaj)?;
    Ok(tensor)
}

pub fn silu(
    tensor: &TypedTensor<f32>,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    let one = TypedTensor::<f32>::from_vec_col_major(vec![], vec![1.0])?;
    let denominator = tensor.neg(backend)?.exp(backend)?.add(&one, backend)?;
    let tensor = tensor.div(&denominator, backend)?;
    Ok(tensor)
}

pub fn gelu(
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

pub fn layernorm(
    tensor: &TypedTensor<f32>,
    weight: &TypedTensor<f32>,
    bias: &TypedTensor<f32>,
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // tensor [x, y, c, n]
    // weight [c]
    // bias   [c]

    let num_x = tensor.shape()[0];
    let num_y = tensor.shape()[1];
    let num_c = tensor.shape()[2];
    let num_n = tensor.shape()[3];
    let count = TypedTensor::<f32>::from_vec_col_major(vec![], vec![num_c as f32])?;
    let epsilon = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.00001f32])?;
    let weight = weight.reshape(&[1, 1, num_c, 1], backend)?; // [1, 1, c, 1]
    let bias = bias.reshape(&[1, 1, num_c, 1], backend)?; // [1, 1, c, 1]

    // mean(tensor, dims = 3)
    // ∑ x / count
    let tensor_mean = tensor
        .reduce_sum(&[2], backend)?
        .div(&count, backend)?
        .reshape(&[num_x, num_y, 1, num_n], backend)?; // [x, y, 1, n]

    // tensor .- tensor_mean
    let tensor_residual = tensor.sub(&tensor_mean, backend)?; // [x, y, c, n]

    // var(tensor, corrected = false, dims = 3)
    // ∑ (x - ⟨x⟩)² / count
    let tensor_var = tensor_residual
        .mul(&tensor_residual, backend)?
        .reduce_sum(&[2], backend)?
        .div(&count, backend)?
        .reshape(&[num_x, num_y, 1, num_n], backend)?; // [x, y, 1, n]

    // √(tensor_var .+ epsilon)
    let denominator = tensor_var.add(&epsilon, backend)?.sqrt(backend)?; // [x, y, 1, n]

    // (tensor .- tensor_mean) ./ √(tensor_var .+ epsilon) .* weight .+ bias
    let tensor = tensor_residual
        .div(&denominator, backend)?
        .mul(&weight, backend)?
        .add(&bias, backend)?; // [x, y, c, n]

    Ok(tensor)
}

pub fn groupnorm(
    tensor: &TypedTensor<f32>,
    weight: &TypedTensor<f32>,
    bias: &TypedTensor<f32>,
    num_g: usize, // Number of groups
    backend: &mut CpuBackend,
) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // tensor [x, y, c, n]
    // weight [c]
    // bias   [c]

    let num_x = tensor.shape()[0];
    let num_y = tensor.shape()[1];
    let num_c = tensor.shape()[2];
    let num_n = tensor.shape()[3];
    let num_i = num_c / num_g; // Number of indices inside a group
    let count =
        TypedTensor::<f32>::from_vec_col_major(vec![], vec![(num_x * num_y * num_i) as f32])?;
    // Notice this differs from layernorm!
    let epsilon = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.00001f32])?;
    let tensor = tensor.reshape(&[num_x, num_y, num_i, num_g, num_n], backend)?; // [x, y, i, g, n]
    let weight = weight.reshape(&[1, 1, num_i, num_g, 1], backend)?; // [1, 1, i, g, 1]
    let bias = bias.reshape(&[1, 1, num_i, num_g, 1], backend)?; // [1, 1, i, g, 1]

    // mean(tensor, dims = 3)
    // ∑ x / count
    let tensor_mean = tensor // [x, y, i, g, n]
        .reduce_sum(&[0, 1, 2], backend)? // [1, 1, g, n] // Notice this differs from layernorm as well!
        .div(&count, backend)? // [1, 1, g, n]
        .reshape(&[1, 1, 1, num_g, num_n], backend)?; // [1, 1, 1, g, n]

    // tensor .- tensor_mean
    let tensor_residual = tensor.sub(&tensor_mean, backend)?; // [x, y, i, g, n]

    // var(tensor, corrected = false, dims = 3)
    // ∑ (x - ⟨x⟩)² / count
    let tensor_var = tensor_residual // [x, y, i, g, n]
        .mul(&tensor_residual, backend)? // [x, y, i, g, n]
        .reduce_sum(&[0, 1, 2], backend)? // [1, 1, g, n]
        .div(&count, backend)? // [1, 1, g, n]
        .reshape(&[1, 1, 1, num_g, num_n], backend)?; // [1, 1, 1, g, n]

    // √(tensor_var .+ epsilon)
    let denominator = tensor_var.add(&epsilon, backend)?.sqrt(backend)?; // [1, 1, 1, g, n]

    // (tensor .- tensor_mean) ./ √(tensor_var .+ epsilon) .* weight .+ bias
    let tensor = tensor_residual // [x, y, i, g, n]
        .div(&denominator, backend)? // [x, y, i, g, n]
        .mul(&weight, backend)? // [x, y, i, g, n]
        .add(&bias, backend)? // [x, y, i, g, n]
        .reshape(&[num_x, num_y, num_c, num_n], backend)?; // [x, y, c, n]

    Ok(tensor)
}

pub fn upsample(tensor: &TypedTensor<f32>) -> Result<TypedTensor<f32>, Box<dyn Error>> {
    // tensor [x, y, c, n]

    let num_x = tensor.shape()[0];
    let num_y = tensor.shape()[1];
    let num_c = tensor.shape()[2];
    let num_n = tensor.shape()[3];

    let colmaj = {
        let mut colmaj = Vec::with_capacity((2 * num_x) * (2 * num_y) * num_c * num_n);
        for n in 0..num_n {
            for c in 0..num_c {
                for new_y in 0..(2 * num_y) {
                    for new_x in 0..(2 * num_x) {
                        let x = new_x / 2;
                        let y = new_y / 2;
                        colmaj.push(*tensor.get(&[x, y, c, n])?);
                    }
                }
            }
        }
        colmaj
    };

    let tensor =
        TypedTensor::<f32>::from_vec_col_major(vec![2 * num_x, 2 * num_y, num_c, num_n], colmaj)?;

    Ok(tensor)
}

pub fn self_attention(
    tensor: &TypedTensor<f32>,
    in_weight: &TypedTensor<f32>,
    out_weight: &TypedTensor<f32>,
    out_bias: &TypedTensor<f32>,
    num_h: usize, // Number of heads
    backend: &mut CpuBackend,
) -> Result<(), Box<dyn Error>> {
    // The model doesn't use in_bias anywhere so I won't support it here.
    // (in other words, in_bias is a zero-vector)

    // d:                     0 <= d < 3*num_c
    // i: index inside a head 0 <= i < num_c/num_h

    // tensor     [x, y, c, n]
    // in_weight  [d, c]
    // out_weight [c, c]
    // out_bias   [c, c]

    /*
    show(&tensor)?;
    println!();
    show(&in_weight)?;
    println!();
    show(&out_weight)?;
    println!();
    show(&out_bias)?;
    println!();
    */

    let num_x = tensor.shape()[0];
    let num_y = tensor.shape()[1];
    let num_c = tensor.shape()[2];
    let num_n = tensor.shape()[3];
    let num_i = num_c / num_h;

    let tensor = [in_weight, tensor].einsum("dc,xycn->xynd", backend)?; // [x, y, n, d]

    // TODO: Maybe I have to consider using views or slices. tenferro supports them.
    let mut chunks = tensor.host_data()?.chunks(num_x * num_y * num_n * num_c);
    let q = TypedTensor::<f32>::from_vec_col_major(
        vec![num_x, num_y, num_n, num_i, num_h],
        chunks.next().unwrap().to_vec(),
    )?; // [x, y, n, i, h]
    let k = TypedTensor::<f32>::from_vec_col_major(
        vec![num_x, num_y, num_n, num_i, num_h],
        chunks.next().unwrap().to_vec(),
    )?; // [X, Y, n, i, h] (I'll call them X and Y so that they match with `einsum`.)
    let v = TypedTensor::<f32>::from_vec_col_major(
        vec![num_x, num_y, num_n, num_i, num_h],
        chunks.next().unwrap().to_vec(),
    )?; // [x, y, n, i, h]

    let tensor = [&k, &q].einsum("XYnih,xynih->XYxynh", backend)?; // [X, Y, x, y, n, h]

    let sqrt_d = TypedTensor::<f32>::from_vec_col_major(vec![], vec![(num_i as f32).sqrt()])?;
    let tensor = tensor.div(&sqrt_d, backend)?; // [X, Y, x, y, n, h]

    let tensor = softmax(

    show(&tensor)?;

    Ok(())
}
