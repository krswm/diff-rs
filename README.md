# Stable Diffusion Inference with tenferro

I built an inference engine for Stable Diffusion from scratch in Rust.

Stable Diffusion is a machine learning model that is trained to generate images from text.

It is built on [tenferro](https://github.com/tensor4all/tenferro-rs), a Rust-native tensor library.

I also built:

- [An inference engine for Stable Diffusion in Julia](https://github.com/krswm/diff-jl)
- [An inference engine for GPT-2 in Rust](https://github.com/krswm/slope-rs)
- [An inference engine for GPT-2 in Julia](https://github.com/krswm/slope-jl)

# Quickstart

I made this project just for **educational purpose**.
Use at your own risk.

It is assumed that you have Git, cURL, and Cargo installed on your machine.

**Clone this repository.**

```
git clone https://github.com/krswm/diff-rs
cd diff-rs
```

**Download the pre-trained Stable Diffusion model from Hugging Face.**

```
mkdir model
curl --location https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/reslove/main/v1-5-pruned.safetensors --output model/model.safetensors
curl --location https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json --output model/vocab.json
curl --location https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt --output model/merges.txt
curl --location https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/text_encoder/config.json --output model/config.json
```

**Compile the Rust code.**

```
cargo build --release
```

**Generate an image.**

```
target/release/diff-rs model output.ppm 'a cat with a hat' ''
```

The third parameter (`'a cat with a hat'` here) is the *positive* prompt.
The model tries to make the generated image look *more* like what the positive prompt describes.

The fourth parameter (`''` here, an empty prompt) is the *negative* prompt.
The model tries to make the generated image look *less* like what the positive prompt describes.
You can leave it empty (`''`), as this example does.

It may take some minutes to finish inference.

The AI-generated image is saved to the path on the second parameter (`output.ppm` here).
It is a PPM image format file.
macOS’s Preview supports the format.
You can convert it to another format using an external program.
For instance, you can convert a PPM image to a PNG image with ImageMagick:

```
magick output.ppm output.png
```

# Credits

- [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239) for devising DDPM.
- [*Convolution as Matrix Multiplication*](https://github.com/alisaaalehi/convolution_as_multiplication) and [*Implement convolutions as matrix operations using im2col*](https://numb3r33.github.io/experiments/convolution/math/deeplearning/2023/12/23/im2col.html) for teaching me how to implement 2D convolution with a matrix-multiplication-based algorithm.
- [*Stable Diffusion implemented from scratch in PyTorch*](https://github.com/hkproj/pytorch-stable-diffusion) for teaching me the technical aspect of the model and providing me a reference implementation.
- [tenferro](https://github.com/tensor4all/tenferro-rs) for providing me an amazing tensor library for Rust. I would not even start this project if tenferro were not released.

# Development

I was interested in the diffusion models because of their application in science.
So, I decided to implement an inference engine of a diffusion model from scratch.
Although its application is not for science, I chose Stable Diffusion because:

- The model weight was openly available.
- I could find many technical information of it online.
- My goal was to understand the internal structure of a diffusion model, so any diffusion model sufficed.

I am looking forward to implement something scientific based on this project in the future.

- 2026-08-31: I started this project.
- 2026-09-28: I finished implementing the inference engine.

This is a hobby project of mine I started from scratch.
I enjoyed a lot by working on this project!

I used [the model weight of Stable Diffusion](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and [a reference implementation (*Stable Diffusion implemented from scratch in PyTorch*)](https://github.com/hkproj/pytorch-stable-diffusion) only for the purpose to observe their behavior as diffusion model architecture.
Except for that, I did **not** use generative AI for this project at all.
