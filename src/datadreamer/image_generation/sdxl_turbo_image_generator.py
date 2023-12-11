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
        generate_image(prompt, negative_prompt, prompt_objects): Generates an image based on the provided prompt.
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
        base = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        base.enable_model_cpu_offload()

        return base

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_objects: Optional[List[str]] = None,
    ) -> Image.Image:
        """Generates an image using the Stable Diffusion Turbo model based on the
        provided prompt.

        Args:
            prompt (str): The positive prompt to guide image generation.
            negative_prompt (str): The negative prompt to avoid certain features in the image.
            prompt_objects (Optional[List[str]]): Optional list of objects to be used in CLIP model testing.

        Returns:
            Image.Image: The generated image.
        """
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
        ).images[0]

        return image

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache."""
        self.base = self.base.to("cpu")
        if self.use_clip_image_tester:
            self.clip_image_tester.release()
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()
