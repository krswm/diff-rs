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

fn main() -> Result<(), Box<dyn Error>> {
    Ok(())
}
