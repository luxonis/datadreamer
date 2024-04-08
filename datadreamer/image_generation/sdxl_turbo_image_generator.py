from __future__ import annotations

from typing import List, Optional

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

from datadreamer.image_generation.image_generator import ImageGenerator


class StableDiffusionTurboImageGenerator(ImageGenerator):
    """A subclass of ImageGenerator specifically designed to use the Stable Diffusion
    Turbo model for faster image generation.

    Attributes:
        base (AutoPipelineForText2Image): The Stable Diffusion Turbo model for image generation.

    Methods:
        _init_gen_model(): Initializes the Stable Diffusion Turbo model.
        generate_images_batch(prompts, negative_prompt, prompt_objects): Generates a batch of images based on the provided prompts.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the StableDiffusionTurboImageGenerator with the given
        arguments."""
        super().__init__(*args, **kwargs)
        self.base = self._init_gen_model()

    def _init_gen_model(self):
        """Initializes the Stable Diffusion Turbo model for image generation.

        Returns:
            AutoPipelineForText2Image: The initialized Stable Diffusion Turbo model.
        """
        if self.device == "cpu":
            print("Loading SDXL Turbo on CPU...")
            base = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                # variant="fp16",
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
            base.to("cpu")
        else:
            print("Loading SDXL Turbo on GPU...")
            base = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            base.enable_model_cpu_offload()

        return base

    def generate_images_batch(
        self,
        prompts: List[str],
        negative_prompt: str,
        prompt_objects: Optional[List[List[str]]] = None,
        batch_size: int = 1,
    ) -> List[Image.Image]:
        """Generates a batch of images using the Stable Diffusion Turbo model based on
        the provided prompts.

        Args:
            prompts (List[str]): A list of positive prompts to guide image generation.
            negative_prompt (str): The negative prompt to avoid certain features in the image.
            prompt_objects (Optional[List[List[str]]]): Optional list of objects for each prompt for CLIP model testing.
            batch_size (int): The number of images to generate in each batch.

        Returns:
            List[Image.Image]: A list of generated images.
        """
        images = self.base(
            prompt=prompts,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
        ).images

        return images

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache."""
        self.base = self.base.to("cpu")
        if self.use_clip_image_tester:
            self.clip_image_tester.release()
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import os

    # Create the generator
    image_generator = StableDiffusionTurboImageGenerator(
        seed=42,
        use_clip_image_tester=False,
        image_tester_patience=1,
        batch_size=8,
        device="cpu",
    )
    prompts = [
        "A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.",
    ] * 16
    prompt_objects = [["aeroplane", "bicycle"]] * 16

    image_paths = []
    counter = 0
    for generated_images_batch in image_generator.generate_images(
        prompts, prompt_objects
    ):
        for generated_image in generated_images_batch:
            image_path = os.path.join("./", f"image_turbo_{counter}.jpg")
            generated_image.save(image_path)
            image_paths.append(image_path)
            counter += 1

    image_generator.release(empty_cuda_cache=True)
