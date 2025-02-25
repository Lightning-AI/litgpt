# Multi-Head Latent Attention (MLA)

## Overview
This document outlines the modifications made to the codebase in the `litgpt` repository to add support for Multi-Head Latent Attention (MLA) block from [DeepSeekV2](https://arxiv.org/abs/2405.04434).

## Changes Made
1. **Configuration**: Added `latent_attention: Optional[bool] = False` parameter to the configuration file to enable the MLA block.
2. **MLA module**: Implemented the MLA module as a separate component in the `litgpt` codebase.
3. **KVCacheCompressed**: Added support for the `KVCacheCompressed` class to store the key-value pairs for the MLA block.
4. **Model**: Modified the GPT model to include the **MLA block** as an alternative component based on the configuration parameter `latent_attention`.
5. **Training**: Updated the training script to support the MLA block and added support for training with the new configuration file `config_hub/pretrain/cfg.yaml`.

## Installation
Follow the updated installation instructions in the `README.md` file.

## Usage
1. **Configuration**: Set the `latent_attention` parameter to `True` in the configuration file to enable the MLA block.
2. **Training**: Run the training script with the updated configuration file.
    ```bash
    litgpt pretrain --config config_hub/pretrain/cfg.yaml
    ```
3. **Inference**: Use the trained model for inference as follows:
    ```bash
    litgpt generate out/pretrain/mla/final/
    ```

## Results
Results are available at [this link](https://docs.google.com/spreadsheets/d/1-VnTDoK5JuNPGMjory_z1hQkI7y-RgiTpTsUpa3bVEg/edit?usp=sharing).

The results highlight that MQA and GQA considerably reduce memory usage and increase the speed of training. However, this comes at the cost of a significant decrease in performance compared to the baseline model.

The MLA block demonstrates a better trade-off between memory usage, speed, and performance. It shows a slight drop in performance compared to the baseline model, while also reducing memory usage. This also comes with a slight increase in training and inference speed. Smaller projection dimensions have been tested for the MLA block, showing a consistent reduction of memory usage but with a significant drop in performance.

Overall, results are not as significant as expected due to the small scale of the model (limited by the GPU memory) and the short training time (~10k steps). Further experiments on larger models, bigger datasets, and longer training are expected to highlight the benefits of the MLA block. Also, further experiments with layer normalization and other hyperparameters are expected to improve the performance of the MLA block.

## Notes
- Pythia was used as model for the experiments because it comes with many versions at different scales.
- `pythia-160m` (160M parameters) was the largest model that could be trained on a single GPU with 16GB memory. 
- For the same reason, the `tinystories` dataset was used for the experiments and the models were trained for only 100M tokens (~10k steps).
- Experiments on larger models, bigger datasets, and longer training are expected to further highlight the benefits of the MLA block.
- All the tested implementations use FlashAttention (as implemented in torch) by default.
- The resulting implementation of MLA depends on the `litgpt` codebase (especially the `CausalSelfAttention` class).
- The implementation of the MLA block is based on the DeepSeekV2 paper and includes support for KV caching (`KVCacheCompressed`) and decoupled RoPE (`apply_rope_mla`).
- A further improvement would be to optimize the implementation for speed and memory usage (for example, by merging matrices at inference like in LoRA). 
    > Fortunately, due to the associative law of matrix multiplication, we can absorb $𝑊^{𝑈𝐾}$ into $𝑊^{𝑈𝑄}$ , and $𝑊^{𝑈𝑉}$ into $𝑊^{𝑂}$. Therefore, we do not need to compute keys and values out for each query. Through this optimization, we avoid the computational overhead for recomputing $k^C_t$ and $v^𝐶_𝑡$ during inference.

    Unfortunately, this was not implemented due to time constraints.
