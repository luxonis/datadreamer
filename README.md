![basereduce Logo](images/dalle-logo.png)

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
- **Edge Model Training**: Train efficient small-scale neural networks for edge deployment.

## Installation

To install `basereduce` from source, follow these steps:

```bash
# Clone the repository
git clone https://github.com/luxonis/basereduce.git
cd basereduce

# Install the package
pip install -e .
```

## Usage

### Overview
The pipelines/generate_dataset_from_scratch.py script is designed to generate and annotate images based on specified object names. This tool uses advanced models to create images and annotate them with bounding boxes for the objects of interest.

### Requirements
Python 3.x
Relevant Python libraries: matplotlib, torch, PIL, numpy, argparse
Pre-installed basereduce package with its submodules: prompt_generation, image_generation, dataset_annotation

### Usage
To use the script, run the following command in your terminal:

```bash
basereduce --save_dir <directory> --object_names <objects> --prompts_number <number>
```

### Parameters
--save_dir (required): Path to the directory where the generated images and their annotations will be saved.
--object_names (required): Space-separated list of object names to be included in the image generation and annotation process. For example, aeroplane bicycle bird.
--prompts_number (optional): The number of prompts to generate for each object. Defaults to 10 if not specified.

### Example
```bash
basereduce --save_dir path/to/save_directory --class_names person moon robot lightsaber tractor --prompts_number 20 --prompt_generator simple --num_objects_range 2 4 --image_generator sdxl-turbo
```
This command will generate images for the objects 'aeroplane', 'bicycle', 'bird', 'boat', and 'person'. The images and their annotations will be saved in /path/to/save_directory.

### Output
Generated images will be saved in the specified save_dir.
Bounding box annotations for each image will be saved in a subdirectory within save_dir, named bboxes.
Additional data including boxes, labels, and object names will be saved in JSON format in the save_dir.

### Note
Ensure that all dependencies are correctly installed and that the basereduce package is properly set up in your Python environment before running the script.

## License

## Acknowledgements

This library was made possible by the use of several open-source projects, including Transformers, Diffusers, and others listed in the requirements.txt.
