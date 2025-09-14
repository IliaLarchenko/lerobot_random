# This code is based on and should be added to the https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py


# ...

import openpi.policies.lekiwi_policy as lekiwi_policy

# ...

@dataclasses.dataclass(frozen=True)
class LeRobotLeKiwiDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    """

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform simply remaps key names here.
        # I keep it here as an example and to align naming with the original notation
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/images/top": "observation.images.top",
                        "observation/images/wrist": "observation.images.wrist",
                        "observation/images/front": "observation.images.front",
                        "observation/state": "observation.state",
                        "action": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        data_transforms = _transforms.Group(
            inputs=[lekiwi_policy.LeKiwiInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[lekiwi_policy.LeKiwiOutputs()],
        )

        # My LeKiwi actions dim is 9, first 5 are joints that should be converted to delta actions.
        # 6th is gripper that should be left unchanged.
        # 7-9 are mobile based velocities and they are 0 in my case anyways.
        delta_action_mask = _transforms.make_bool_mask(5, -4)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


_CONFIGS = [
    # ...
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_lekiwi_lora",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", action_horizon=30),

        data=LeRobotLeKiwiDataConfig(
            repo_id="IliaLarchenko/vla_demo",
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        
        # In my case, 5k steps was enough
        num_train_steps=5_000,

        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", action_horizon=30
        ).get_freeze_filter(),

        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_lekiwi",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", action_horizon=30),

        data=LeRobotLeKiwiDataConfig(
            repo_id="IliaLarchenko/vla_demo",
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        
        # In my case, 5k steps was enough
        num_train_steps=5_000,
    ),

    TrainConfig(
        # Change the name to reflect your model and dataset.
        # In my experience LoRA fine tuning of FAST model was not working well I would recommend to use full fine-tuning instead.
        name="pi0_lekiwi_fast_lora",

        # Action dim should be aligned with the robot and dataset, 9 for LeKiwi
        # Action horizon is shorter than in action expert predictions, I kept the rest as default.
        model=pi0_fast.Pi0FASTConfig(action_dim=9, paligemma_variant="gemma_2b_lora", action_horizon=15, max_token_len=180),

        data=LeRobotLeKiwiDataConfig(
            repo_id="IliaLarchenko/vla_demo",
            base_config=DataConfig(prompt_from_task=True),
        ),

        # Use the pi0-FAST base model checkpoint.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),

        # Theoretically it should converge faster (x5 in the paper) but it was not the case in my experience
        num_train_steps=5_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=9, paligemma_variant="gemma_2b_lora", action_horizon=10, max_token_len=180).get_freeze_filter(),

        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_lekiwi_fast",

        # Action dim should be aligned with the robot and dataset, 9 for LeKiwi
        # Action horizon is shorter than in action expert predictions, I kept the rest as default.
        model=pi0_fast.Pi0FASTConfig(action_dim=9, paligemma_variant="gemma_2b", action_horizon=15, max_token_len=180),

        data=LeRobotLeKiwiDataConfig(
            repo_id="IliaLarchenko/vla_demo",
            base_config=DataConfig(prompt_from_task=True),
        ),

        # Use the pi0-FAST base model checkpoint.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),

        # Theoretically it should converge faster (x5 in the paper) but it was not the case in my experience
        num_train_steps=5_000,
    ),

    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi05_lekiwi",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=15),

        data=LeRobotLeKiwiDataConfig(
            repo_id="IliaLarchenko/vla_demo",
            base_config=DataConfig(prompt_from_task=True),
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),        
        num_train_steps=5_000,
    ),

    
    # ...
]

# ...