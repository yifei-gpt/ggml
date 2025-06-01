# ggml

[Roadmap](https://github.com/users/ggerganov/projects/7) / [Manifesto](https://github.com/ggerganov/llama.cpp/discussions/205)

Tensor library for machine learning

***Note that this project is under active development. \
Some of the development is currently happening in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repos***

## Features

- Low-level cross-platform implementation
- Integer quantization support
- Broad hardware support
- Automatic differentiation
- ADAM and L-BFGS optimizers
- No third-party dependencies
- Zero memory allocations during runtime

## Build

```bash
git clone https://github.com/ggml-org/ggml
cd ggml

# build the examples
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```



## Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [The GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
