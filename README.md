> [!WARNING] This repository only works with old version of lerobot, before they introduced the processors. 
> The newer version is jet to be implemented.

# LeRobot UNet MeanFlow

This repository implements the MeanFlow objective for a policy with a **U-Net backbone**.

Check the original [MeanFlow paper](https://arxiv.org/abs/2505.13447) for more information about the algorithm.

## Overview

This implementation combines:
- **U-Net Backbone**: Convolutional encoder-decoder from DiffusionPolicy
- **MeanFlow Loss**: Efficient one-step inference from DiTMeanFlow

With MeanFlow, only **one step inference** is required during deployment, making it significantly faster than traditional diffusion policies.

## Getting Started

### Installation

```bash
git clone https://github.com/danielsanjosepro/lerobot_policy_unetmeanflow.git
cd lerobot_policy_unetmeanflow

# Install with pixi (recommended)
pixi shell  # to activate the environment
```

### Training

Train on PushT environment:

```bash
python -m lerobot.scripts.train --config_path config/train_unetmeanflow_pusht.json
```


## Related Work

- [DiT Flow](https://github.com/danielsanjosepro/lerobot_policy_ditflow) - Transformer-based implementation with a Flow Matching objective
- [DiT MeanFlow](https://github.com/danielsanjosepro/lerobot_policy_ditmeanflow) - Transformer-based implementation with a MeanFlow objective
- [DiT Flow](https://github.com/huggingface/lerobot/pull/680) - Original DiT Flow implementation
- [MP1](https://github.com/LogSSim/MP1) -  MP1 : Mean Flow Tames Policy Learning in 1-step for Robotic Manipulation
- [MeanFlow Paper](https://arxiv.org/abs/2505.13447) - Original algorithm paper

