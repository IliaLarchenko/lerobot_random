This repository is a collection of random useful things for LeRobot, LeKiwi, and SO-ARM100/101. I experiment with these robots and library a lot and will share some useful pieces of code or stl models here.

# Scripts

[`setup_remote.sh`](scripts/setup_remote.sh) - a script to set up a remote machine to use LeRobot and other tools that includes most of the dependencies that I needed in practice and some extra useful tools. Just run `bash setup_remote.sh` to install all the dependencies - it usually takes up to 10 minutes (depends on the internet speed), and should fix most of the issues.

# Code 

VLA section contains some files useful for experiments with VLAs (like Pi0, Pi05, Gr00t, SmolVLA, etc.). Refer to the [README](vla/README.md) for more details.

# 3D models

[`servo_wheel_hub.stl`](stl/servo_wheel_hub.stl): is a slightly modified version of the [original LeKiwi wheel hub](https://github.com/SIGRobotics-UIUC/LeKiwi/blob/main/3DPrintMeshes/servo_wheel_hub.stl). I altered it so it is possible to use the screws included into the servo motors and omni wheels. I used [these wheels](https://www.alibaba.com/product-detail/14049-Zoty-100mm-Omnidirectional-Wheel-Omni_62514267136.html) - that is one of the recommended models.

`camera_mount_n.stl` [[1](stl/camera_mount_1.stl), [2](stl/camera_mount_2.stl), [3](stl/camera_mount_3.stl), [4](stl/camera_mount_4.stl)] - different mounts for [this camera](https://s.lazada.co.th/s.viOCK). I use it with custom gripper, not all of them will work with the original so-100.

[`realsense_d435_mount.stl`](stl/realsense_d435_mount.stl) - top view mount for the RealSense D435 depth camera.

[`column.stl`](stl/column.stl), [`column_small.stl`](stl/column_small.stl), [`column_small_back.stl`](stl/column_small_back.stl) - support columns for the upper level of modified LeKiwi. More details [here](https://www.youtube.com/watch?v=lJ6PH--el54).

# Links
- [LeRobot](https://github.com/huggingface/LeRobot)
- [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi)
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)