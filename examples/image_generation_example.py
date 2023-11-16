from basereduce.image_generation.image_generator import (
    StableDiffusionImageGenerator,
    GenModelName,
)
import matplotlib.pyplot as plt

# Define some prompts for image generation
prompts = [
    "a futuristic city skyline at sunset",
    "a portrait of a robotic dog in a meadow",
    "an astronaut riding a horse on the moon",
]

# Initialize the image generator with the required parameters
image_generator = StableDiffusionImageGenerator(
    seed=42.0,
    model_name=GenModelName.STABLE_DIFFUSION_XL,
    prompt_prefix="A high-resolution image of ",
    prompt_suffix=" with intricate details.",
    negative_prompt="a low-quality image, blurry details, incorrect anatomy",
)

# Generate images using the list of prompts
generated_images = image_generator.generate_images(prompts)

# Optionally test each image and do something with the results
for img in generated_images:
    if image_generator._test_image(img):
        plt.imshow(img)
        plt.axis("off")  # Hide the axis
        plt.show()
    else:
        # Handle the case for invalid images
        print("Generated image did not pass the test.")
