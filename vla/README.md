This folder contains some files useful for work with VLAs (Pi0/Pi0.5, Gr00t N.1.5, SmolVLA) using LeRobot library.

Each model authors provide good documentation and examples and following them should be enough to get started.

Here I share some extra scripts and configs that can be useful. This repository accompanies my video series about VLAs. You can find it on YouTube:
[![Ep.1 Introduction to VLAs](https://img.youtube.com/vi/8dZUOo5xWFw/0.jpg)](https://youtu.be/8dZUOo5xWFw)


# Pi0

<!-- [![Ep.2 Pi 0](https://img.youtube.com/vi/8dZUOo5xWFw/0.jpg)](https://youtu.be/8dZUOo5xWFw) -->

To fine tune Pi0 you can follow instruction in the official repository: https://github.com/Physical-Intelligence/openpi

I share extra files:

- `config.py` - contains a class `LeRobotLeKiwiDataConfig` and 3 train configs: for Pi0 and Pi0-FAST (full and LoRA) and Pi05.
They should be added to the https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
You can use them as extra examples and adapt for your robots and datasets.

- `lekiwi_policy.py` - contains a class `LeKiwiInputs` and `LeKiwiOutputs` that are used to convert inputs and outputs to the model to the expected format. Should be added to the https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies
You can use them as extra examples and adapt for your robots and datasets.

- `evaluate_pi0.py` - a wrapper that allows to use Pi0 policy to control LeKiwi robot using LeRobot library.
You need to adjust your robot IP and ID to use it.
You can specify prompt (task description) and number of actions to execute for each prediction.


# Troubleshooting

Below are some issues that I faced during training and inference, I hope they will be useful for you.

## Dependencies

Most of the issues that I faced were related to some missing dependencies. I created a script to set up a remote machine to use LeRobot and other tools that includes most of the dependencies that I needed in practice and some extra useful tools. You can find it in the [`scripts/setup_remote.sh`](../scripts/setup_remote.sh) file. Just run `bash setup_remote.sh` to install all the dependencies - it usually takes around 10 minutes, and should fix most of the issues.

## FFmpeg

In my experience especially often issues were related to ffmpeg version. If you see one of the following errors it can be it:

`ValueError: No valid stream found in input file. Is -1 of the desired media type?`

`RuntimeError: Could not load libtorchcodec.`

Make sure that you have version 7.1+

```bash
ffmpeg -version
```

To fix follow the official lerobot instruction create a new conda environment and install ffmpeg 7.1+:

```bash
conda create -n lerobot -c conda-forge python=3.11 ffmpeg
```

### Video backend for Gr00t

Don't forget to set `video_backend = "torchvision_av"` for lerobot dataset when training Gr00t N1.5 (this issue can have similar symptoms as in the previous section)

If needed install `av` library:
`pip install -U av`

## Tokens decoding in Pi0-FAST

`Error decoding tokens: cannot reshape array of size N into shape (M)`

This issue can arise in the beginning of the training of Pi-FAST model while it is still learning the proper decoding.
Try to train the model longer - it should fix the issue.

I also observed that it happens much more often when fine-tuning the model with LoRA, and I would recommend to use full fine-tuning for the FAST model instead.

## Lerobot imports issue when evaluating Pi0

Currently openpi is using old version of lerobot and it can cause some issues with imports because my evaluation wrapper is using the new version.

If you see an error like:
`ModuleNotFoundError: No module named 'lerobot.robots'`

The lerobot version in your environment is too old for the wrapper.

If you see an error like this:
`ModuleNotFoundError: No module named 'lerobot.common'`

The lerobot version in your environment is too new for the openpi.

It will be fixed as soon as openpi updates lerobot import, but the easiest way to fix it is to install the latest version of lerobot and remove `common` form all imports like `lerobot.common` in openpi. Generally you need to change it in only one file: `src/openpi/training/data_loader.py`


## FlashAttention issue in Gr00t

I faced the following error during inference of Gr00t model on 2080Ti GPU:

`RuntimeError: FlashAttention only supports Ampere GPUs or newer.`

Gr00t uses FlashAttention which is not supported on 2080Ti and older GPUs.
To overcome it you will need some code changes. Please refer to this pull request: https://github.com/NVIDIA/Isaac-GR00T/pull/150


