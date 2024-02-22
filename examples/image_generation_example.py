import matplotlib.pyplot as plt

from datadreamer.image_generation import (
    StableDiffusionTurboImageGenerator,
)
from datadreamer.image_generation.clip_image_tester import ClipImageTester

# Define some prompts for image generation
prompts = [
    "a futuristic city skyline at sunset",
    "a robotic dog in a meadow",
    "an army of astronauts riding a horse on the moon",
]

prompt_objects = [
    ["city", "skyline"],
    ["robotic dog"],
    ["astronauts", "horse", "moon"],
]

# Initialize the image generator with the required parameters
image_generator = StableDiffusionTurboImageGenerator(
    prompt_prefix="A photo of ",
    seed=42.0,
)

# Generate images using the list of prompts assuming yield is used in the generator
generated_images = []
for generated_images_batch in image_generator.generate_images(prompts, prompt_objects):
    generated_images.extend(generated_images_batch)

# Release the model and empty the CUDA cache
image_generator.release(empty_cuda_cache=True)

# Initialize the image tester
image_tester = ClipImageTester()


# Optionally test each image and do something with the results
for img, prompt, prompt_objs in zip(generated_images, prompts, prompt_objects):
    # Test the image against the prompt objects
    passed, probs, num_passed = image_tester.test_image(img, prompt_objs)
    print(f"Image passed test for prompt '{prompt}': {passed}")
    print(f"Prompt objects: {prompt_objs}")
    print(f"Object probabilities: {probs}")
    print(f"Number of objects passed: {num_passed}")

    # Show the image
    plt.imshow(img)
    plt.show()
