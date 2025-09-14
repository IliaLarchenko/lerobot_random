# Based on https://github.com/Physical-Intelligence/openpi/src/openpi/policies/libero_policy.py

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(9),
        "observation/images/front": np.random.randint(256, size=(640, 360, 3), dtype=np.uint8),
        "observation/images/wrist": np.random.randint(256, size=(640, 360, 3), dtype=np.uint8),
        "observation/images/top": np.random.randint(256, size=(640, 360, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LeKiwiInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    """

    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        ## I have 3 cameras and it is not perfectly fit the type of cameras Pi0 uses 
        ## So I pass the front image instead of the right wrist image - it still works
        base_image = _parse_image(data["observation/images/top"])
        wrist_image = _parse_image(data["observation/images/wrist"])
        front_image = _parse_image(data["observation/images/front"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": front_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "action" in data:
            # We are padding to the model action dim.
            actions = transforms.pad_to_dim(data["action"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LeKiwiOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 9 actions -- since we padded actions above to fit the model action
        return {"actions": np.asarray(data["actions"][:, :9])}
