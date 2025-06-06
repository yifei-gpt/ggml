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
git clone https://github.com/yifei-gpt/ggml.git
cd ggml

# build the examples
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```
## Transformer Block
Transformer_Block code is located in the:  
```bash
/examples/simple/transformer_block.cpp
```

After building, running it like this:  
```bash
cd build
./bin/transformer_block <n_tokens> <n_embd> <n_head_q> <n_head_kv> <ffn_hidden_dim> <batch_size> <dtype> <N_INPUTS> <threads>
```

###  Argument Description

| Argument           | Description                                                      |
|--------------------|------------------------------------------------------------------|
| `<n_tokens>`       | Number of tokens per input sequence                              |
| `<n_embd>`         | Embedding dimension                                              |
| `<n_head_q>`       | Number of attention heads for queries                            |
| `<n_head_kv>`      | Number of attention heads for keys/values (GQA support)          |
| `<ffn_hidden_dim>` | Hidden layer size in the feed-forward network (FFN)              |
| `<batch_size>`     | Number of samples per batch                                      |
| `<dtype>`          | Weight type: `fp32`, `fp16`, `int8`, or `int4`                   |
| `<N_INPUTS>`       | Total number of input sequences to process                       |
| `<threads>`        | Threads                                                          |

For example,
```bash
./bin/transformer_block 512 2048 32 8 8192 1 int8 71 4    ##LLaMA 3.2-1B
./bin/transformer_block 512 3072 24 8 8192 1 int8 71 4   ##LLaMA 3.2-3B
./bin/transformer_block 512 4096 32 8 14336 1 int8 71 4    ##LLaMA 3-8B
```



## Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [The GGUF file format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
