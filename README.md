![basereduce Logo](images/grid_image_3x2_generated_dataset.jpg)

# BaseReduce

`basereduce` is a toolkit that allows you to train edge AI models, even if you don't have any data to start with! Here's what makes BaseReduce special:

- **Train Without Data**: You don't need a huge collection of images or data to train your AI. With BaseReduce, you can create a synthetic dataset from scratch using powerful AI that knows how to generate images.
  
- **Extract Knowledge from Base Computer Vision models**: There's a lot of knowledge locked away in big, pre-trained AI models. BaseReduce knows how to tap into these "Foundation models" and transfer their knowledge to your smaller model, making it wise beyond its size.

- **Small but Mighty Models**: Our goal is to help you create models that are small enough to fit into any device but still perform like the big ones in the special tasks they were trained for.

And why the name BaseReduce? It's simple:

- **Base**: We start with the basics, the "base" knowledge from large AI models.
- **Reduce**: We then "reduce" that knowledge into a smaller, more compact form that's easy to handle.

So, in a nutshell, BaseReduce helps you go from zero to AI hero, building intelligent models efficiently and effectively.

## Features

- **Prompt Generation**: Automate the creation of image prompts using powerful language models.
- **Image Generation**: Generate synthetic datasets with state-of-the-art generative models.
- **Dataset Annotation**: Leverage foundation models to automatically label datasets.
- **Edge Model Training**: Train efficient small-scale neural networks for edge deployment. (not part of this library)

## Installation

To install `basereduce` from source, follow these steps:

```bash
# Clone the repository
git clone https://github.com/luxonis/basereduce.git
cd basereduce

# Install the package
pip install -e .
```

## Hardware Requirements
To ensure optimal performance and compatibility with the libraries used in this project, the following hardware specifications are recommended:

- `GPU`: A CUDA-compatible GPU with a minimum of 16 GB memory. This is essential for libraries like `torch`, `torchvision`, `transformers`, and `diffusers`, which leverage CUDA for accelerated computing in machine learning and image processing tasks.
- `RAM`: At least 16 GB of system RAM, although more (32 GB or above) is beneficial for handling large datasets and intensive computations.

## Usage

### Overview
The `basereduce/pipelines/generate_dataset_from_scratch.py` (`basereduce` command) script is a powerful tool for generating and annotating images with specific objects. It uses advanced models to both create images and accurately annotate them with bounding boxes for designated objects.

### Usage
Run the following command in your terminal to use the script:

```bash
basereduce --save_dir <directory> --class_names <objects> --prompts_number <number> [additional options]
```

### Main Parameters
- `--save_dir` (required): Path to the directory for saving generated images and annotations.
- `--class_names` (required): Space-separated list of object names for image generation and annotation. Example: person moon robot.
- `--prompts_number` (optional): Number of prompts to generate for each object. Defaults to 10.

### Additional Parameters

- `--task`: Choose between `detection` and `classification`. Default is `detection`.
- `--num_objects_range`: Range of objects in a prompt. Default is 1 to 3.
- `--prompt_generator`: Choose between `simple` and `lm` (language model). Default is `simple`.
- `--image_generator`: Choose image generator, e.g., `sdxl` or `sdxl-turbo`. Default is `sdxl-turbo`.
- `--image_annotator`: Specify the image annotator, like `owlv2`. Default is `owlv2`.
- `--conf_threshold`: Confidence threshold for object detection. Default is 0.15.
- `--use_tta`: Toggle test time augmentation for object detection. Default is True.
- `--enhance_class_names`: Enhance class names with synonyms. Default is False.
- `--use_image_tester`: Use image tester for image generation. Default is False.
- `--image_tester_patience`: Patience level for image tester. Default is 1.
- `--device`: Choose between `cuda` and `cpu`. Default is cuda.
- `--seed`: Set a random seed for image generation. Default is 42.

### Example
```bash
basereduce --save_dir path/to/save_directory --class_names person moon robot --prompts_number 20 --prompt_generator simple --num_objects_range 2 4 --image_generator sdxl-turbo
```
This command generates images for the specified objects, saving them and their annotations in the given directory. The script allows customization of the generation process through various parameters, adapting to different needs and hardware configurations.

### Output
The dataset comprises two primary components: images and their corresponding annotations, stored as JSON files.

```bash

save_dir/
│
├── image_1.jpg
├── image_2.jpg
├── ...
├── image_n.jpg
├── prompts.json
└── detection_annotations.json/classification_annotations.json
```

### Annotations Format
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



### Note
Ensure that all dependencies are correctly installed and that the basereduce package is properly set up in your Python environment before running the script.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This library was made possible by the use of several open-source projects, including Transformers, Diffusers, and others listed in the requirements.txt.
