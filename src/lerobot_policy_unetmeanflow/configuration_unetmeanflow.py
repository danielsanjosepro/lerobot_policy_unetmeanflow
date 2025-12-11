#!/usr/bin/env python

from dataclasses import dataclass, field

from typing_extensions import override

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.constants import ACTION
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("unetmeanflow")
@dataclass
class UnetMeanFlowConfig(PreTrainedConfig):
    """Configuration class for UnetMeanFlowPolicy.

    This configuration merges the U-Net architecture from DiffusionPolicy with the meanflow algorithm
    from DiTMeanFlow. It combines the convolutional U-Net backbone with meanflow's dual time parameters
    and gradient-based velocity prediction.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        # Inherited from DiffusionConfig - core structure
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Model action prediction size.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.

        # Inherited from DiffusionConfig - vision processing
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone.
        crop_is_random: Whether the crop should be random at training time.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder for each camera view.

        # Inherited from DiffusionConfig - U-Net architecture
        down_dims: Feature dimension for each stage of temporal downsampling in the U-Net.
        kernel_size: The convolutional kernel size of the U-Net.
        n_groups: Number of groups used in the group norm of the U-Net's convolutional blocks.
        diffusion_step_embed_dim: The U-Net is conditioned on timesteps via a small non-linear network.
        use_film_scale_modulation: Whether to use scale modulation in addition to bias modulation for FiLM.

        # From DiTMeanFlowConfig - meanflow algorithm
        use_meanflow: Whether to use meanflow algorithm (vs standard flow matching). Default: True (FIXED).
        use_adaptative_loss: Whether to use adaptive loss weighting or MSE loss. Default: True (FIXED).
        use_autograd_functional_jvp: Whether to use torch.autograd.functional.jvp instead of torch.func.jvp.
        flow_ratio: Ratio of sampled ts and rs that are the same (resulting in standard flow setting).
        time_distribution: Distribution to use for sampling t and r ("uniform" or "lognorm").
        log_norm_mu: Mean for log-normal time distribution.
        log_norm_sigma: Standard deviation for log-normal time distribution.

        # From DiTMeanFlowConfig - inference
        do_multi_step_sampling: Whether to use multi-step sampling at inference.
        inference_timesteps: Number of timesteps for multi-step sampling.

        # From DiTMeanFlowConfig - classifier-free guidance
        cfg_prob: Probability of using unconditional samples for classifier-free guidance.
        cfg_omega: Weight for classifier-free guidance.

        # Training and optimization
        clip_sample: Whether to clip samples during training/inference.
        clip_sample_range: Range for sample clipping.
        do_mask_loss_for_padding: Whether to mask loss for padded actions.
    """

    # Inputs / output structure (from DiffusionConfig)
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # FIXED: Change ACTION normalization to MEAN_STD (was MIN_MAX)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MEAN_STD,  # FIXED: was MIN_MAX
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Vision backbone (from DiffusionConfig)
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = False
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # U-Net architecture (from DiffusionConfig)
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Meanflow algorithm parameters (from DiTMeanFlowConfig)
    # FIXED: Set use_meanflow to True by default (was False)
    use_meanflow: bool = True  # FIXED: was False

    # Time sampling parameters
    time_distribution: str = "uniform"  # "uniform" or "lognorm"
    log_norm_mu: float = -0.4
    log_norm_sigma: float = 1.0

    # Loss parameters
    # FIXED: Set use_adaptative_loss to True by default (was False)
    use_adaptative_loss: bool = True  # FIXED: was False
    flow_ratio: float = 0.5  # Ratio of samples where t == r (standard flow)

    # FIXED: Add use_autograd_functional_jvp parameter (was missing)
    use_autograd_functional_jvp: bool = False

    # Inference parameters
    do_multi_step_sampling: bool = (
        False  # Use one-step sampling by default for meanflow
    )
    inference_timesteps: int = 1  # FIXED: was 100, should be 1 for one-step

    # Classifier-free guidance parameters
    cfg_prob: float = 0.1
    cfg_omega: float = 2.0  # If < 1.0, no classifier-free guidance is used

    # Sample clipping
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 0.5
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    features_to_exclude: list[str] = field(default_factory=lambda: [])

    # NOTE: if you set features to include, features_to_exclude is ignored.
    features_to_include: list[str] | None = None

    @override
    def __post_init__(self):
        super().__post_init__()

        # Input validation from DiffusionConfig
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

        # Input validation from DiTMeanFlowConfig
        if self.time_distribution not in ["uniform", "lognorm"]:
            raise ValueError(
                f"time_distribution {self.time_distribution} not supported."
            )

        if not (0.0 <= self.flow_ratio <= 1.0):
            raise ValueError(
                f"flow_ratio {self.flow_ratio} must be between 0.0 and 1.0."
            )

        if self.cfg_omega < 1.0:
            raise ValueError(
                f"cfg_omega {self.cfg_omega} must be greater than or equal to 1.0."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if self.features_to_include is not None:
            assert ACTION in self.features_to_include, (
                f"`{ACTION}` must be included in `features_to_include`. Got {self.features_to_include}."
            )
            for feature in self.features_to_include:
                if (
                    feature not in self.input_features
                    and feature not in self.output_features
                ):
                    raise ValueError(
                        f"Feature '{feature}' in `features_to_include` not found in input or output features."
                        f" Available input features: {list(self.input_features.keys())}. "
                        f"Available output features: {list(self.output_features.keys())}."
                    )

            # Create features_to_exclude from features_to_include
            all_features = list(self.input_features.keys()) + list(
                self.output_features.keys()
            )
            self.features_to_exclude = [
                feature
                for feature in all_features
                if feature not in self.features_to_include
            ]

        for feature_to_exclude in self.features_to_exclude:
            if feature_to_exclude in self.input_features:
                del self.input_features[feature_to_exclude]
            elif feature_to_exclude in self.output_features:
                del self.output_features[feature_to_exclude]
            else:
                raise ValueError(
                    f"Feature '{feature_to_exclude}' not found in input or output features."
                    f" Available input features: {list(self.input_features.keys())}. "
                    f"Available output features: {list(self.output_features.keys())}."
                )

        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if (
                    self.crop_shape[0] > image_ft.shape[1]
                    or self.crop_shape[1] > image_ft.shape[2]
                ):
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        if self.image_features:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def use_classifier_free_guidance(self) -> bool:
        """Whether to use classifier-free guidance or not."""
        return self.cfg_omega >= 1.0 and self.cfg_prob > 0.0

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
