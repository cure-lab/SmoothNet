 # SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos (ECCV 2022)

This repo is the official implementation of "**SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos**". 
[[Paper]](https://arxiv.org/abs/2112.13715)  [[Project]](https://ailingzeng.site/smoothnet)

## Update
- [x] Support SmoothNet in [MMPose](https://github.com/open-mmlab/mmpose) [Release v0.25.0](https://github.com/open-mmlab/mmpose/releases/tag/v0.25.0) and [MMHuman3D](https://github.com/open-mmlab/mmhuman3d) as a smoothing strategy!

- [x] Clean version is released! 
- [x] To further improve SmoothNet as a near online smoothing strategy, we reduce the original window size 64 to **32** frames by default! 
- [x] We also provide the pretrained models with the window size 8, 16, 32 and 64 frames [here](https://drive.google.com/drive/folders/1AsOm10AReDKt4HSVAQ0MsZ1Fp-_18IV3?usp=sharing). 

It currently includes **code, data, log and models** for the following tasks: 
- 2D human pose estimation
- 3D human pose estimation
- Body recovery via a SMPL model

### Major Features

- Model training and evaluation for **2D pose, 3D pose, and SMPL body representation**
- Supporting **6 popular datasets** ([AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), [Sub-JHMDB](http://jhmdb.is.tue.mpg.de/), [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/), [MuPoTS-3D](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/), [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)) and providing cleaned estimation results of **13 popular pose estimation backbones**([SPIN](https://github.com/nkolot/SPIN), [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE), [VIBE](https://github.com/mkocabas/VIBE), [CPN](https://github.com/chenyilun95/tf-cpn), [FCN](https://github.com/una-dinosauria/3d-pose-baseline), [Hourglass](http://www-personal.umich.edu/~alnewell/pose), [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), [RLE](https://github.com/Jeff-sjtu/res-loglikelihood-regression), [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), [TposeNet](https://github.com/vegesm/pose_refinement), [EFT](https://github.com/facebookresearch/eft), [PARE](https://pare.is.tue.mpg.de/), [SimplePose](https://github.com/microsoft/human-pose-estimation.pytorch))

## Description

When analyzing human motion videos, the output jitters from existing pose estimators are highly-unbalanced with varied estimation errors across frames. Most frames in a video are relatively easy to estimate and only suffer from slight jitters. In contrast, for rarely seen or occluded actions, the estimated positions of multiple joints largely deviate from the ground truth values for a consecutive sequence of frames, rendering significant jitters on them.

To tackle this problem, we propose to attach **a dedicated temporal-only refinement network** to existing pose estimators for jitter mitigation, named SmoothNet. Unlike existing learning-based solutions that employ spatio-temporal models to co-optimize per-frame precision and temporal smoothness at all the joints, SmoothNet models the natural smoothness characteristics in body movements by learning the long-range temporal relations of every joint without considering the noisy correlations among joints. With a simple yet effective motion-aware fully-connected network, SmoothNet improves the temporal smoothness of existing pose estimators significantly and enhances the estimation accuracy of those challenging frames as a side-effect. Moreover, as a temporal-only model, a unique advantage of SmoothNet is its strong transferability across various types of estimators and datasets. Comprehensive experiments on five datasets with eleven popular backbone networks across 2D and 3D pose estimation and body recovery tasks demonstrate the efficacy of the proposed solution. Our code and datasets are provided in the supplementary materials.


## Results

SmoothNet is a plug-and-play post-processing network to smooth any outputs of existing pose estimators. To fit well across datasets, backbones, and modalities with lower MPJPE and PA-MPJPE, we provide **THREE pre-trained models** (Train on [AIST-VIBE-3D](configs/aist_vibe_3D.yaml), [3DPW-SPIN-3D](configs/pw3d_spin_3D.yaml), and [H36M-FCN-3D](configs/h36m_fcn_3D.yaml)) to handle all existing issues. 

Please refer to our supplementary materials to check the cross-model validation in detail. Noted that all models can obtain **lower and similar Accels** than the compared backbone estimators. The differences are in MPJPEs and PA-MPJPEs.

**Due to the temporal-only network without spatial modelings, SmoothNet is trained on 3D position representations only, and can be tested on 2D, 3D, and 6D representations, respectively.**

### 3D Keypoint Results

| Dataset | Estimator | MPJPE (Input/Output):arrow_down: | Accel (Input/Output):arrow_down: | Pretrain model |
| ------- | --------- | ------------------ | ------------------ | ------------ |
| AIST++    | SPIN      | 107.17/95.21            | 33.19/4.17           | [checkpoint](https://drive.google.com/file/d/101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK/view?usp=sharing) / [config](configs/aist_vibe_3D.yaml) |
| AIST++   | TCMR*       | 106.72/105.51            | 6.4/4.24           | [checkpoint](https://drive.google.com/file/d/101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK/view?usp=sharing) / [config](configs/aist_vibe_3D.yaml)|
| AIST++    | VIBE*       | 106.90/97.47            | 31.64/4.15          | [checkpoint](https://drive.google.com/file/d/101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK/view?usp=sharing) / [config](configs/aist_vibe_3D.yaml)|
| Human3.6M    | FCN       |  54.55/52.72        | 19.17/1.03       |   [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M    | RLE       |  48.87/48.27              | 7.75/0.90          |  [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M    | TCMR*       |  73.57/73.89              | 3.77/2.79          |  [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M    | VIBE*       |  78.10/77.23              | 15.81/2.86          |  [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M    | Videopose(T=27)*       |  50.13/50.04             | 3.53/0.88          |  [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M    | Videopose(T=81)*       |  48.97/48.89             | 3.06/0.87          |  [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M    | Videopose(T=243)*       |  48.11/48.05             | 2.82/0.87          |  [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| MPI-INF-3DHP    | SPIN       |  100.74/92.89             | 28.54/6.54          |  [checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| MPI-INF-3DHP    | TCMR*       |  92.83/88.93             | 7.92/6.49          |  [checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| MPI-INF-3DHP    | VIBE*       |  92.39/87.57             | 22.37/6.5          |  [checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| MuPoTS    | TposeNet*      | 103.33/100.78            | 12.7/7.23           | [checkpoint](https://drive.google.com/file/d/101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK/view?usp=sharing) / [config](configs/aist_vibe_3D.yaml) |
| MuPoTS    | TposeNet+RefineNet*      | 93.97/91.78            | 9.53/7.21           | [checkpoint](https://drive.google.com/file/d/101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK/view?usp=sharing) / [config](configs/aist_vibe_3D.yaml) |
| 3DPW    | EFT       |  90.32/88.40             | 32.71/6.07          |  [checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| 3DPW    | EFT       |  90.32/86.39             | 32.71/6.30          |  [checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)(additional training)|
| 3DPW    | PARE      |  78.91/78.11             | 25.64/5.91          |  [checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| 3DPW    | SPIN      |  96.85/95.84             | 34.55/6.17          |  [checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| 3DPW    | TCMR*      |  86.46/86.48             | 6.76/5.95          |  [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| 3DPW    | VIBE*      |  82.97/81.49             | 23.16/5.98          | [checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|

### 2D Keypoint Results


| Dataset | Estimator | MPJPE (Input/Output):arrow_down: | Accel (Input/Output):arrow_down: | Pretrain model |
| ------- | --------- | ------------------ | ------------------ | ------------ |
| Human3.6M   | CPN      | 6.67/6.45            | 2.91/0.14           |[checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M   | Hourglass      | 9.42/9.25            | 1.54/0.15           |[checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M   | HRNet      | 4.59/4.54            | 1.01/0.13           |[checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| Human3.6M   | RLE      | 5.14/5.11            | 0.9/0.13           |[checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|


### SMPL Results


| Dataset | Estimator | MPJPE (Input/Output):arrow_down: | Accel (Input/Output):arrow_down: | Pretrain model |
| ------- | --------- | ------------------ | ------------------ | ------------ |
| AIST++   | SPIN      | 107.72/103.00            | 33.21/5.72           |[checkpoint](https://drive.google.com/file/d/101TH_Z8uiXD58d_xkuFTh5bI4NtRm_cK/view?usp=sharing) / [config](configs/aist_vibe_3D.yaml) |
| AIST++   | TCMR*      | 106.95/106.39            | 6.47/4.68           |[checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| AIST++   | VIBE*      | 107.41/102.06            | 31.65/5.95           |[checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| 3DPW   | EFT      | 91.60/89.57            | 33.38/7.89           |[checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| 3DPW   | PARE      | 79.93/78.68            | 26.45/6.31           |[checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| 3DPW   | SPIN      | 99.28/97.81            | 34.95/7.40           |[checkpoint](https://drive.google.com/file/d/106MnTXFLfMlJ2W7Fvw2vAlsUuQFdUe6k/view?usp=sharing) / [config](configs/pw3d_spin_3D.yaml)|
| 3DPW   | TCMR*      | 88.46/88.37            | 7.12/6.52           |[checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|
| 3DPW   | VIBE*      | 84.27/83.14            | 23.59/7.24           |[checkpoint](https://drive.google.com/file/d/1ZketGlY4qA3kFp044T1-PaykV2llNUjB/view?usp=sharing) / [config](configs/h36m_fcn_3D.yaml)|

* \* means the used pose estimators are using temporal information. 
* The usage of SmoothNet for better performance: a SOTA **single-frame** estimator (e.g., PARE) + SmoothNet
* Since TCMR uses a sliding window method to smooth the poses, which causes over-smoothness issue, SmoothNet will be hard to further decrease the MPJPE, PA-MPJPE.


## Getting Started

### Environment Requirement

SmoothNet has been implemented and tested on Pytorch 1.10.1 with python >= 3.6. It supports both GPU and CPU inference. 

Clone the repo:
```bash
git clone https://github.com/cure-lab/SmoothNet.git
```

We recommend you prepare the environment using `conda`:
```bash
# conda
source scripts/install_conda.sh
```

### Prepare Data

All the data used in our experiment can be downloaded here. 

[Google Drive](https://drive.google.com/drive/folders/19Cu-_gqylFZAOTmHXzK52C80DKb0Tfx_?usp=sharing)

[Baidu Netdisk](https://pan.baidu.com/s/1J6EV4uwThcn-W_GNuc4ZPw?pwd=eb5x)

The sructure of the repository should look like this:

```
|-- configs
    |-- aist_vibe_3D.yaml
    |-- ...
|-- data
    |-- checkpoints         # pretrained checkpoints
    |-- poses               # cleaned detected poses and groundtruth poses
    |-- smpl                # SMPL parameters
|-- lib
    |-- core
        |-- ...
    |-- dataset
        |-- ...
    |-- models
        |-- ...
    |-- utils
        |-- ...
|-- results                 # folders including log files, checkpoints, running config and tensorboard logs
|-- scripts
    |-- install_conda.sh
|-- eval_smoothnet.py       # SmoothNet evaluation
|-- train_smoothnet.py      # SmoothNet training
|-- README.md
|-- LICENSE
|-- requirements.txt
```

If you want to add your own dataset, please follow these steps (noted that this is also how the provided data is organized):

1. Organize your data into corresponding type according to the body representation. The file structure is shown as follows:

    ```
    |-- [your dataset]\_[estimator]\_[2D/3D/smpl]
        |-- detected
            |-- [your dataset]\_[estimator]\_[2D/3D/smpl]_test.npz
            |-- [your dataset]\_[estimator]\_[2D/3D/smpl]_train.npz
        |-- groundtruth
            |-- [your dataset]\_gt\_[2D/3D/smpl]_test.npz
            |-- [your dataset]\_gt\_[2D/3D/smpl]_train.npz
    ```
    It is fine if you only have training or testing data. The content in each .npz file is consist of "imgname" and "human poses", which is related to the body representation you use.
    
    - 3D keypoints: 

        - imgname: Strings containing the image and sequence name with format [sequence_name]/[image_name](string "" if the sequence_name and image_name not available). 
        - keypoints_3d: 3D joint position. The shape of each sequence is corresponding_sequence_length*(keypoints_number*3). The order of it is the same with imgname

    - 2D keypoints

        - imgname: Strings containing the image and sequence name with format [sequence_name]/[image_name](string "" if the sequence_name and image_name not available). 
        - keypoints_2d: 2D joint position. The shape of each sequence is corresponding_sequence_length*(keypoints_number*2). The order of it is the same with imgname

    - SMPL

        - imgname: Strings containing the image and sequence name with format [sequence_name]/[image_name](string "" if the sequence_name and image_name not available). 
        - pose: pose parameters. The shape of each sequence is corresponding_sequence_length*72. The order of it is the same with imgname
        - shape: shape parameters. The shape of each sequence is corresponding_sequence_length*10. The order of it is the same with imgname
 
2. If you use 3D keypoints as the body representation, add the root of all keypoints ``cfg.DATASET.ROOT_[your dataset]_[estimator]_3D`` in [evaluate_config.py](lib/core/evaluate_config.py), [train_config.py](lib/core/train_config.py) or [visualize_config.py](lib/core/visualize_config.py) according to your purpose(test, train or visualize).
3. Construct your own dataset following [the existing dataset files](lib/dataset). You might need to modify the detailed implementation depending on your data characteristics.

### Training

Run the commands below to start training:

```shell script
python train_smoothnet.py --cfg [config file] --dataset_name [dataset name] --estimator [backbone estimator you use] --body_representation [smpl/3D/2D] --slide_window_size [slide window size]
```

For example, you can train on 3D representation of Human3.6M using backbone estimator FCN with silde window size 8 by:

```shell script
python train_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 8
```

You can easily train on multiple datasets using "," to split multiple datasets / estimator / body representation. For example, you can train on `AIST++` - `VIBE` - `3D` and `3DPW` - `SPIN` - `3D` with silde window size 8 by:

```shell script
python train_smoothnet.py --cfg configs/h36m_fcn_3D.yaml --dataset_name aist,pw3d --estimator vibe,spin --body_representation 3D,3D  --slide_window_size 8
```

Note that the training and testing datasets should be downloaded and prepared before training.

### Evaluation

Run the commands below to start evaluation:

```shell script
python eval_smoothnet.py --cfg [config file] --checkpoint [pretrained checkpoint] --dataset_name [dataset name] --estimator [backbone estimator you use] --body_representation [smpl/3D/2D] --slide_window_size [slide window size] --tradition [savgol/oneeuro/gaus1d]
```

For example, you can evaluate `MPI-INF-3DHP` - `TCMR` - `3D` and `MPI-INF-3DHP` - `VIBE` - `3D` using SmoothNet trained on `3DPW` - `SPIN` - `3D` with silde window size 8, and compare the results with traditional filters `oneeuro` by:

```shell script
python eval_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoints_8.pth.tar --dataset_name mpiinf3dhp,mpiinf3dhp --estimator tcmr,vibe --body_representation 3D,3D --slide_window_size 8 --tradition oneeuro
```

Note that the pretrained checkpoints and testing datasets should be downloaded and prepared before evaluation.

The data and checkpoints used in our experiment can be downloaded here. 

[Google Drive](https://drive.google.com/drive/folders/19Cu-_gqylFZAOTmHXzK52C80DKb0Tfx_?usp=sharing)

[Baidu Netdisk](https://pan.baidu.com/s/1J6EV4uwThcn-W_GNuc4ZPw?pwd=eb5x)

### Visualization

Here, we only provide demo visualization based on offline processed detected poses of specific datasets(e.g. AIST++, Human3.6M, and 3DPW). To visualize on arbitrary given video, please refer to the [inference/demo](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/getting_started.md) of [MMHuman3D](https://github.com/open-mmlab/mmhuman3d).

un the commands below to start evaluation:

```shell script
python visualize_smoothnet.py --cfg [config file] --checkpoint [pretrained checkpoint] --dataset_name [dataset name] --estimator [backbone estimator you use] --body_representation [smpl/3D/2D] --slide_window_size [slide window size] --visualize_video_id [visualize sequence id] --output_video_path [visualization output video path]
```

For example, you can visualize the `second` sequence of `3DPW` - `SPIN` - `3D` using SmoothNet trained on `3DPW` - `SPIN` - `3D` with silde window size 32, and output the video to `./visualize` by:

```shell script
python visualize_smoothnet.py --cfg configs/pw3d_spin_3D.yaml --checkpoint data/checkpoints/pw3d_spin_3D/checkpoints_8.pth.tar --dataset_name pw3d --estimator spin --body_representation 3D --slide_window_size 32 --visualize_video_id 2 --output_video_path ./visualize
```

## Citing SmoothNet

If you find this repository useful for your work, please consider citing it as follows:

```bibtex 
@inproceedings{zeng2022smoothnet,
      title={SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos},
      author={Zeng, Ailing and Yang, Lei and Ju, Xuan and Li, Jiefeng and Wang, Jianyi and Xu, Qiang},
      booktitle={European Conference on Computer Vision},
      year={2022},
      organization={Springer}
}
```

Please remember to cite all the datasets and backbone estimators if you use them in your experiments.


## License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](./LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](./LICENSE). Third-party datasets and software are subject to their respective licenses.
