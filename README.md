# DataDreamer

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/datadreamer/blob/main/examples/generate_dataset_and_train_yolo.ipynb)
[![Project Video](https://img.shields.io/static/v1?label=Project&message=Video&color=red)](https://www.youtube.com/watch?v=6FcSz3uFqRI)
[![Blog Post](https://img.shields.io/static/v1?label=Blog&message=Post&color=red)](https://discuss.luxonis.com/blog/3272-datadreamer-creating-custom-datasets-made-easy)

![DataDreamer examples](https://raw.githubusercontent.com/luxonis/datadreamer/main/images/grid_image_3x2_generated_dataset.jpg)

<a name="quickstart"></a>

## 🚀 Quickstart

To generate your dataset with custom classes, you need to execute only two commands:

```bash
pip install datadreamer
datadreamer --class_names person moon robot
```

<a name ="overview"></a>

## 🌟 Overview

<img src='https://raw.githubusercontent.com/luxonis/datadreamer/main/images/datadreamer_scheme.gif' align="center">

`DataDreamer` is an advanced toolkit engineered to facilitate the development of edge AI models, irrespective of initial data availability. Distinctive features of DataDreamer include:

- **Synthetic Data Generation**: Eliminate the dependency on extensive datasets for AI training. DataDreamer empowers users to generate synthetic datasets from the ground up, utilizing advanced AI algorithms capable of producing high-quality, diverse images.

- **Knowledge Extraction from Foundational Models**: `DataDreamer` leverages the latent knowledge embedded within sophisticated, pre-trained AI models. This capability allows for the transfer of expansive understanding from these "Foundation models" to smaller, custom-built models, enhancing their capabilities significantly.

- **Efficient and Potent Models**: The primary objective of `DataDreamer` is to enable the creation of compact models that are both size-efficient for integration into any device and robust in performance for specialized tasks.

## 📜 Table of contents

- [🚀 Quickstart](#quickstart)
- [🌟 Overview](#overview)
- [🛠️ Features](#features)
- [💻 Installation](#installation)
- [⚙️ Hardware Requirements](#hardware-requirements)
- [📋 Usage](#usage)
  - [🎯 Main Parameters](#main-parameters)
  - [🔧 Additional Parameters](#additional-parameters)
  - [🤖 Available Models](#available-models)
  - [💡 Example](#example)
  - [📦 Output](#output)
  - [📝 Annotations Format](#annotations-format)
- [⚠️ Limitations](#limitations)
- [📄 License](#license)
- [🙏 Acknowledgements](#acknowledgements)

<a name="features"></a>

## 🛠️ Features

- **Prompt Generation**: Automate the creation of image prompts using powerful language models.

  *Provided class names: \["horse", "robot"\]*

  *Generated prompt: "A photo of a horse and a robot coexisting peacefully in the midst of a serene pasture."*

- **Image Generation**: Generate synthetic datasets with state-of-the-art generative models.

- **Dataset Annotation**: Leverage foundation models to label datasets automatically.

- **Edge Model Training**: Train efficient small-scale neural networks for edge deployment. (not part of this library)

<img src="https://raw.githubusercontent.com/luxonis/datadreamer/main/images/generated_image.jpg" width="400"><img src="https://raw.githubusercontent.com/luxonis/datadreamer/main/images/annotated_image.jpg" width="400">

<a name="installation"></a>

## 💻 Installation

There are two ways to install the `datadreamer` library:

**Using pip**:

To install with pip:

```bash
pip install datadreamer
```

**Using Docker (for Linux/Windows)**:

Pull Docker Image from GHCR:

```bash
docker pull ghcr.io/luxonis/datadreamer:latest
```

Or build Docker Image from source:

```bash
# Clone the repository
git clone https://github.com/luxonis/datadreamer.git
cd datadreamer

# Build Docker Image
docker build -t datadreamer .
```

**Run Docker Container (assuming it's GHCR image, otherwise replace `ghcr.io/luxonis/datadreamer:latest` with `datadreamer`)**

Run on CPU:

```bash
docker run --rm -v "$(pwd):/app" ghcr.io/luxonis/datadreamer:latest --save_dir generated_dataset --device cpu
```

Run on GPU, make sure to have nvidia-docker installed:

```bash
docker run --rm --gpus all -v "$(pwd):/app" ghcr.io/luxonis/datadreamer:latest --save_dir generated_dataset --device cuda
```

These commands mount the current directory ($(pwd)) to the /app directory inside the container, allowing you to access files from your local machine.

<a name="hardware-requirements"></a>

## ⚙️ Hardware Requirements

To ensure optimal performance and compatibility with the libraries used in this project, the following hardware specifications are recommended:

- `GPU`: A CUDA-compatible GPU with a minimum of 16 GB memory. This is essential for libraries like `torch`, `torchvision`, `transformers`, and `diffusers`, which leverage CUDA for accelerated computing in machine learning and image processing tasks.
- `RAM`: At least 16 GB of system RAM, although more (32 GB or above) is beneficial for handling large datasets and intensive computations.

<a name="usage"></a>

## 📋 Usage

The `datadreamer/pipelines/generate_dataset_from_scratch.py` (`datadreamer` command) script is a powerful tool for generating and annotating images with specific objects. It uses advanced models to both create images and accurately annotate them with bounding boxes for designated objects.

Run the following command in your terminal to use the script:

```bash
datadreamer --save_dir <directory> --class_names <objects> --prompts_number <number> [additional options]
```

<a name="main-parameters"></a>

### 🎯 Main Parameters

- `--save_dir` (required): Path to the directory for saving generated images and annotations.
- `--class_names` (required): Space-separated list of object names for image generation and annotation. Example: `person moon robot`.
- `--prompts_number` (optional): Number of prompts to generate for each object. Defaults to `10`.
- `--annotate_only` (optional): Only annotate the images without generating new ones, prompt and image generator will be skipped. Defaults to `False`.

<a name="additional-parameters"></a>

### 🔧 Additional Parameters

- `--task`: Choose between detection and classification. Default is `detection`.
- `--dataset_format`: Format of the dataset. Defaults to `raw`. Supported values: `raw`, `yolo`, `coco`, `luxonis-dataset`.
- `--split_ratios`: Split ratios for train, validation, and test sets. Defaults to `[0.8, 0.1, 0.1]`.
- `--num_objects_range`: Range of objects in a prompt. Default is 1 to 3.
- `--prompt_generator`: Choose between `simple`, `lm` (language model) and `tiny` (tiny LM). Default is `simple`.
- `--image_generator`: Choose image generator, e.g., `sdxl`, `sdxl-turbo` or `sdxl-lightning`. Default is `sdxl-turbo`.
- `--image_annotator`: Specify the image annotator, like `owlv2` for object detection or `clip` for image classification. Default is `owlv2`.
- `--conf_threshold`: Confidence threshold for annotation. Default is `0.15`.
- `--annotation_iou_threshold`: Intersection over Union (IoU) threshold for annotation. Default is `0.2`.
- `--prompt_prefix`: Prefix to add to every image generation prompt. Default is `""`.
- `--prompt_suffix`: Suffix to add to every image generation prompt, e.g., for adding details like resolution. Default is `", hd, 8k, highly detailed"`.
- `--negative_prompt`: Negative prompts to guide the generation away from certain features. Default is `"cartoon, blue skin, painting, scrispture, golden, illustration, worst quality, low quality, normal quality:2, unrealistic dream, low resolution,  static, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, bad anatomy"`.
- `--use_tta`: Toggle test time augmentation for object detection. Default is `True`.
- `--synonym_generator`: Enhance class names with synonyms. Default is `none`. Other options are `llm`, `wordnet`.
- `--use_image_tester`: Use image tester for image generation. Default is `False`.
- `--image_tester_patience`: Patience level for image tester. Default is `1`.
- `--lm_quantization`: Quantization to use for Mistral language model. Choose between `none` and `4bit`. Default is `none`.
- `--annotator_size`: Size of the annotator model to use. Choose between `base` and `large`. Default is `base`.
- `--batch_size_prompt`: Batch size for prompt generation. Default is 64.
- `--batch_size_annotation`: Batch size for annotation. Default is `8`.
- `--batch_size_image`: Batch size for image generation. Default is `1`.
- `--device`: Choose between `cuda` and `cpu`. Default is `cuda`.
- `--seed`: Set a random seed for image and prompt generation. Default is `42`.

<a name="available-models"></a>

### 🤖 Available Models

| Model Category    | Model Names                                                                           | Description/Notes                       |
| ----------------- | ------------------------------------------------------------------------------------- | --------------------------------------- |
| Prompt Generation | [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | Semantically rich prompts               |
|                   | [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Tiny LM                                 |
|                   | Simple random generator                                                               | Joins randomly chosen object names      |
| Image Generation  | [SDXL-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)           | Slow and accurate (1024x1024 images)    |
|                   | [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo)                           | Fast and less accurate (512x512 images) |
|                   | [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)                     | Fast and accurate (1024x1024 images)    |
| Image Annotation  | [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)                    | Open-Vocabulary object detector         |
|                   | [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)                           | Zero-shot-image-classification          |

<a name="example"></a>

### 💡 Example

```bash
datadreamer --save_dir path/to/save_directory --class_names person moon robot --prompts_number 20 --prompt_generator simple --num_objects_range 1 3 --image_generator sdxl-turbo
```

This command generates images for the specified objects, saving them and their annotations in the given directory. The script allows customization of the generation process through various parameters, adapting to different needs and hardware configurations.

<a name="output"></a>

### 📦 Output

The dataset comprises two primary components: images and their corresponding annotations, stored as JSON files.

```bash

save_dir/
│
├── image_1.jpg
├── image_2.jpg
├── ...
├── image_n.jpg
├── prompts.json
└── annotations.json
```

<a name="annotations-format"></a>

### 📝 Annotations Format

1. Detection Annotations (detection_annotations.json):

- Each entry corresponds to an image and contains bounding boxes and labels for objects in the image.
- Format:

```bash
{
  "image_path": {
    "boxes": [[x_min, y_min, x_max, y_max], ...],
    "labels": [label_index, ...]
  },
  ...
  "class_names": ["class1", "class2", ...]
}
```

2. Classification Annotations (classification_annotations.json):

- Each entry corresponds to an image and contains labels for the image.
- Format:

```bash
{
  "image_path": {
    "labels": [label_index, ...]
  },
  ...
  "class_names": ["class1", "class2", ...]
}
```

<a name="limitations"></a>

## ⚠️ Limitations

While the datadreamer library leverages advanced Generative models to synthesize datasets and Foundation models for annotation, there are inherent limitations to consider:

- `Incomplete Object Representation`: Occasionally, the generative models might not include all desired objects in the synthetic images. This could result from the complexity of the scene or limitations within the model's learned patterns.

- `Annotation Accuracy`: The annotations created by foundation computer vision models may not always be precise. These models strive for accuracy, but like all automated systems, they are not infallible and can sometimes produce erroneous or ambiguous labels. However, we have implemented several strategies to mitigate these issues, such as Test Time Augmentation (TTA), usage of synonyms for class names and careful selection of the confidence/IOU thresholds.

Despite these limitations, the datasets created by datadreamer provide a valuable foundation for developing and training models, especially for edge computing scenarios where data availability is often a challenge. The synthetic and annotated data should be seen as a stepping stone, granting a significant head start in the model development process.

<a name="license"></a>

## 📄 License

This project is licensed under the [Apache License, Version 2.0](https://opensource.org/license/apache-2-0/) - see the [LICENSE](LICENSE) file for details.

The above license does not cover the models. Please see the license of each model in the table above.

<a name="acknowledgements"></a>

## 🙏 Acknowledgements

This library was made possible by the use of several open-source projects, including Transformers, Diffusers, and others listed in the requirements.txt.

[SD-XL 1.0 License](https://github.com/Stability-AI/generative-models/blob/main/model_licenses/LICENSE-SDXL1.0)
[SDXL-Turbo License](https://github.com/Stability-AI/generative-models/blob/main/model_licenses/LICENSE-SDXL-Turbo)
