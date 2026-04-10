

## Notice

This virtual try-on task is based on **VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization** (published in **CVPR 2021**). Since the official paper does not provide training code, we did not train the model ourselves. Instead, we used the pretrained model to perform clothing replacement. We also implemented the preprocessing steps including clothing segmentation, human parsing, and pose estimation.

To simplify testing, the system only requires two input images: one clothing image and one person image (both with background). These inputs are automatically converted into the six required inputs of the original VITON-HD model, as shown below:

| input         | Description                                      |
| ------------- | ------------------------------------------------ |
| cloth         | Clothing image with background removed           |
| cloth_mask    | Binary mask of clothing (white clothes on black) |
| person        | Person image with background removed             |
| person_parse  | Semantic segmentation of human body parts        |
| openpose_img  | Pose keypoints visualization (color-coded)       |
| openpose_json | Pose keypoints coordinates in JSON format        |

## Installation

#### VITON-HD

Windows environment:

| Windows | 10       |
| ------- | -------- |
| GPU     | RTX 3070 |
| cudnn   | 7.6.5    |
| Cuda    | 3.7.10   |

Conda virtual environment:

| python      | 3.8.18       |
| ----------- | ------------ |
| cudatoolkit | 8.0          |
| torch       | 1.13.0+cu117 |
| torchaudio  | 0.13.0+cu117 |
| torchvision | 0.14.0+cu117 |

#### Preprocess

##### 1. Clothing Segmentation

We use a pretrained model from U2-Net. No additional dependencies are required.

##### 2. Human Parsing

We use the [AILIA](https://github.com/axinc-ai/ailia-models) model library. Python 3.6 or newer is required.


```
pip3 install ailia
```

In addition to the steps in the [Tutorial](https://github.com/axinc-ai/ailia-models/blob/master/TUTORIAL.md), you also need to apply for the ailia SDK from [ailia](https://github.com/axinc-ai/ailia-sdk?tab=readme-ov-file). Alternatively, you can directly apply by clicking [Download a free evaluation version of ailia SDK](https://axip-console.appspot.com/trial/terms/AILIA?lang=en).
You will need to provide a valid email address to receive the download link and the license file. After downloading, move the license file (`AILIA.lic`) to the directory where `bootstrap.py` is located (`ailia_sdk/python`). Then, download the `requirements.txt` file from this page (https://github.com/axinc-ai/ailia-models) into the same directory as `bootstrap.py`, and run the following command to install the required dependencies on Windows, Mac, and Linux:
```
pip install -r requirements.txt
```

**Note: For Jetson and Raspberry Pi installation details, please refer to the [Tutorial](https://github.com/axinc-ai/ailia-models/blob/master/TUTORIAL.md).**

**Note:** The ailia model expires after 30 days. Once expired, you need to reapply for the SDK and repeat the above steps.

##### 3. Pose Estimation

We use the OpenPose implementation from the [Windows Portable Version](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo). Follow the steps in the [Windows Portable Version](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo) to download the release. Then download the OpenPose models from Kaggle: [openpose_model | Kaggle](https://www.kaggle.com/datasets/changethetuneman/openpose-model).

After that, run:

```
models/getBaseModels.bat
models/getCOCO_and_MPII_optional.bat
```

These commands will automatically download the required models. After downloading, place the images to be processed into the `./examples/media/` directory, and run the following command in the root directory:

```
bin\OpenPoseDemo.exe --image_dir examples\media --hand --write_images output\ --write_json output/ --disable_blending
```


This will perform pose estimation on all files in the `./examples/media/` directory.

**Note:** It is recommended to clean the `media` directory beforehand to avoid insufficient GPU memory or long processing time when using CPU.

## Usage

A batch script `run.bat` has already been provided. When running on different machines, simply modify the absolute paths accordingly. Then execute the following command in the terminal:

> run person_00 cloth.jpg

Here, `person_00` is the name of the person image (**without the .jpg or .png extension**), and `cloth.jpg` is the clothing image.

After execution, it will automatically update the person-clothing pairs in `datasets\test_pairs.txt`. The result of the virtual try-on will be saved as "results\test\person_cloth.jpg".



**Note:** Before running, make sure both input images are resized to **768×1024**, otherwise dimension mismatch errors may occur.

## Principle

#### VITON-HD

For training a neural network, the ideal dataset would consist of:
- Input: a person wearing their own clothes and a target clothing image
- Ground truth: the same person wearing the target clothing

However, such datasets are difficult to obtain. In most cases, we only have product images and photos of models wearing those products (i.e., inputs without corresponding labels). Therefore, we adopt the VITON approach, which removes clothing information from the person image.

Since real-world images are often high-resolution, we use VITON-HD, which performs well at higher resolutions. It reduces artifacts caused by misalignment between warped clothing and the target region, producing results at a resolution of 1024×768.

The process is as follows:

1. First, obtain human semantic segmentation and pose maps using existing methods. Then combine them with the original image and remove the clothing and arm regions.
2. Next, use the segmentation image (without clothing and arms) as ground truth. Take the segmentation map (with clothing and arms removed), pose map, and target clothing as input, and use a U-Net-based CGAN to predict the segmentation map of the person wearing the target clothing.
3. Then, use the segmentation (without clothing and arms), pose map, and the predicted segmentation (with target clothing). Extract the clothing region and apply TPS transformation to warp the target clothing so that it better fits the person’s pose.
4. Finally, combine all the above information to generate the final image. Since the warped clothing cannot perfectly align with the original clothing region, misalignment may occur. To address this, the ALIAS (ALIgnment-Aware Segment normalization) module is used to reduce artifacts caused by misalignment. It can also generate skin regions previously occluded by clothing, and accurately reconstruct clothing details such as patterns, styles, and textures. Meanwhile, the original face, hands, pants, and other details of the person are well preserved, resulting in high-quality outputs.
   
![image-20240429194550431](VITON-HD.png)

### Preprocess

#### 1. Clothing Segmentation

##### 1.1 Clothing Segmentation

To perform clothing segmentation, we use U2-Net. [U2-Net](https://github.com/xuebinqin/U-2-Net)

Since our target is clothing segmentation, we selected the dataset [[iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data)]. This dataset provides detailed annotations for different parts of clothing, achieving a level of accuracy that exceeds our requirements. For convenience, we chose this dataset.

In the output segmentation map, different parts of the clothing are labeled with different colors. For implementation details, please refer to the official [U2-Net](https://github.com/xuebinqin/U-2-Net) repository. The model I trained on my laptop performs reasonably well, but the repository [Clothes Segmentation using U2NET](https://github.com/Charlie839242/cloth-segmentation) provides models trained for more iterations, which yield better results. You can download them if needed.

The original clothing image is shown below:

<img src=".\datasets\test\cloth\cloth.jpg" alt="cloth" width="30%" />

##### 1.2 Convert All Non-Black Pixels to White

Since the segmentation output assigns different colors to different parts of the clothing, while VITON-HD requires a binary (black-and-white) segmentation mask, this conversion step is necessary.

The resulting output is shown below:

<img src=".\datasets\test\cloth-mask\cloth.jpg" alt="cloth" width="30%" />

##### 1.3 Remove the Background of the Clothing Image Using the Binary Mask

The resulting output is shown below:

<img src=".\datasets\test\cloth\cloth.jpg" alt="cloth" width="30%" />

At this point, we have obtained the **cloth** and **cloth_mask** inputs required by VITON-HD.


#### 2. Human Parsing

In the VITON-HD paper, the authors use the segmentation model from ACGPN for human parsing. However, that dataset does not include a label for the neck, while VITON-HD’s segmentation results do include a neck label.

Therefore, we use the ATR training set from [Self-Correction-Human-Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing). In this dataset, the neck and face are labeled with the same color. However, images generated based on the LIP dataset do not include a neck label and only annotate the face. Thus, we subtract the face region obtained from LIP from the combined face+neck region obtained from ATR to isolate the neck region, and then assign it a different color for labeling.

Next, all non-black pixels are converted to white to obtain the segmentation mask, and a similar process as in Section 1.3 is applied to remove the background and obtain the person image.

Since [Self-Correction-Human-Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) only provides GPU-based inference, we instead use another library that integrates multiple AI models: [AILIA](https://github.com/axinc-ai/ailia-models). This library conveniently includes both ATR and LIP models and provides converted ONNX models, allowing inference on CPU.

After segmentation, we found that the color labels differ from those in the VITON-HD dataset. Therefore, we need to convert the colors to match the VITON-HD format.

The color mapping is shown as follows:

| ATR Output Map |                       |             | Sample Map   |                      |             |
| -------------- | --------------------- | ----------- | ------------ | -------------------- | ----------- |
| Palette Index  | Color                 | Body Part   | Palette Index| Color                | Body Part   |
| 0              | [0, 0, 0]: Black      | Background  | 0            | [0, 0, 0]: Black     | Background  |
| 2              | [0, 128, 0]: Green    | Hair        | 2            | [254, 0, 0]: Red     | Hair        |
| 4              | [0, 0, 128]: Blue     | Clothes     | 5            | [254, 85, 0]: Orange | Clothes     |
| 5              | [128, 0, 128]: Purple | Pants       | 9            | [0, 85, 85]: Dark Green | \        |
| 11             | [192, 128, 0]: Brownish Yellow | Face + Neck | 10           | [85, 51, 0]: Brown   | Neck        |
| 12             | [64, 0, 128]: Dark Purple | Right Leg | 12           | [0, 128, 0]: Green   | Pants       |
| 13             | [192, 0, 128]: Pink   | Left Leg    | 13           | [0, 0, 254]: Blue    | Face        |
| 14             | [64, 128, 128]: Light Blue | Right Arm | 14           | [51, 169, 220]: Light Blue | Right Arm |
| 15             | [192, 128, 128]: Skin Color | Left Arm | 15           | [0, 254, 254]: Bright Blue | Left Arm |
|                |                       |             | 16           | [85, 254, 169]: Light Green | Right Leg |
|                |                       |             | 17           | [169, 254, 85]: Bright Green | Left Leg |

For details about the color conversion process, please refer to:
[Charlie839242/An-implementation-of-preprocess-in-VITON-HD-](https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-?tab=readme-ov-file#13-利用获得的黑白分割图来去除衣服图片的背景)

At this point, we have obtained the **person** and **person_parse** inputs required by VITON-HD.

<img src=".\datasets\test\image\person_00.jpg" alt="person_00" width="30%" /><img src=".\datasets\test\image-parse\person_00.png" alt="person_00" width="30%" />

#### 3. Pose Estimation

To generate the pose images and keypoint coordinates required by VITON-HD, we use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). On Windows, we implement it using the [Windows Portable Version](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo). The rendered skeleton images are saved in the `output` directory, while the keypoint coordinates and related information are stored in JSON files.

In this way, we obtain both the pose images and the corresponding keypoint data in JSON format.

<img src=".\datasets\test\openpose-img\person_00_rendered.png" alt="person_00_rendered" width="30%" />

After completing the above preprocessing steps, we obtain all the required inputs for VITON-HD. This enables virtual try-on by simply providing a person image and a target clothing image (both with background). The method achieves good performance even on high-resolution images and effectively reduces artifacts.


## References

- [shadow2496/VITON-HD: Official PyTorch implementation of "VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization" (CVPR 2021) (github.com)](https://github.com/shadow2496/VITON-HD?tab=readme-ov-file)
- [Charlie839242/An-implementation-of-preprocess-in-VITON-HD-: This repository contains the implementations of the preprocessing stages of VITON-HD (github.com)](https://github.com/Charlie839242/An-implementation-of-preprocess-in-VITON-HD-)
- [条件生成对抗网络——cGAN原理与代码 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629503280)
- [2D虚拟试衣2021最新论文 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/366500069)
- [姿势模仿教程 | 🇨🇳中文版 | Petoi Doc Center](https://docs.petoi.com/v/chinese/ying-yong-shi-li/zi-shi-mo-fang-jiao-cheng)
- [ailia-models/TUTORIAL.md at master · axinc-ai/ailia-models (github.com)](https://github.com/axinc-ai/ailia-models/blob/master/TUTORIAL.md)
- [axinc-ai/ailia-sdk: cross-platform high speed inference SDK (github.com)](https://github.com/axinc-ai/ailia-sdk?tab=readme-ov-file)
- [openpose/doc/installation/0_index.md at master · CMU-Perceptual-Computing-Lab/openpose (github.com)](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo)
- [openpose_model | Kaggle](https://www.kaggle.com/datasets/changethetuneman/openpose-model)

