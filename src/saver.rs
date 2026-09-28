// Stable Diffusion Inference with tenferro
// Copyright (C) 2026  Kurosawa Mutsumi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Save a decoded RGB tensor as a NetPPM image.

use std::error::Error;
use std::fs::File;
use std::io::Write;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

pub fn save_as_netppm_image(tensor: &TypedTensor<f32>, path: &str) -> Result<(), Box<dyn Error>> {
    // tensor [x, y, r, n]
    //
    // r: R, G, or B?

    let num_x = tensor.shape()[0];
    let num_y = tensor.shape()[1];
    let num_r = tensor.shape()[2];
    let num_n = tensor.shape()[3];

    assert_eq!(num_r, 3);
    assert_eq!(num_n, 1);

    // -1.0          0.0       0.9921875  <- The elements of the output of decoder (floats)
    //   |------------|------------|
    //   0           128          255     <- are linearly converted to this (integers)

    //               127
    // 0.9921875 is -----
    //               128

    let mut backend = CpuBackend::new();
    let one = TypedTensor::<f32>::from_vec_col_major(vec![], vec![1.0])?;
    let tensor = tensor.add(&one, &mut backend)?;
    let a_hundred_and_twenty_eight = TypedTensor::<f32>::from_vec_col_major(vec![], vec![128.0])?;
    let tensor = tensor.mul(&a_hundred_and_twenty_eight, &mut backend)?;
    let zero = TypedTensor::<f32>::from_vec_col_major(vec![], vec![0.0])?;
    let two_five_five = TypedTensor::<f32>::from_vec_col_major(vec![], vec![255.0])?;
    let tensor = tensor.clamp(&zero, &two_five_five, &mut backend)?;

    let mut file = File::create(path)?;

    // The image is RGB (P3) with width `num_x` and height `num_y`.
    // The range of the value for R, G, and B is 0 to 255 (256 possible values).
    file.write_all(format!("P3 {num_x} {num_y} 255\n").as_bytes())?;

    for y in 0..num_y {
        for x in 0..num_x {
            let r = tensor.get(&[x, y, 0, 0])?;
            let g = tensor.get(&[x, y, 1, 0])?;
            let b = tensor.get(&[x, y, 2, 0])?;
            file.write_all(format!("{r:.0} {g:.0} {b:.0}\n").as_bytes())?;
        }
    }
    
    Ok(())
}
