// Machine Learning Utilities

// I'm going to build smaller parts first.
// What do I need?
// - show
//   Prints the edges of a tensor, for example, if A is 30x40 matrix show A[0, 0], A[29, 0], A[0, 39], and A[29, 39].
//   so that I can compare the computation with reference implementation.
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

#[allow(dead_code)]
pub fn show(tensor: &TypedTensor<f32>) -> Result<(), Box<dyn Error>> {
    if tensor.rank() == 1 {
        let shape = tensor.shape();
        println!("[0] = {}", tensor.get(&[0])?);
        println!("[{}] = {}", shape[0] - 1, tensor.get(&[shape[0] - 1])?);
    }
    if tensor.rank() == 2 {
        let shape = tensor.shape();
        println!("[0, 0] = {}", tensor.get(&[0, 0])?);
        println!("[{}, 0] = {}", shape[0] - 1, tensor.get(&[shape[0] - 1, 0])?);
        println!("[0, {}] = {}", shape[1] - 1, tensor.get(&[0, 0])?);
        println!("[{}, {}] = {}", shape[0] - 1, shape[1] - 1, tensor.get(&[shape[0] - 1, 0])?);
    }
}
