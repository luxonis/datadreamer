from __future__ import annotations

from typing import List, Optional

import torch
from diffusers import DiffusionPipeline
from loguru import logger
from optimum.quanto import freeze, qint8, quantize
from PIL import Image

from datadreamer.image_generation.image_generator import ImageGenerator


class Shuttle3DiffusionImageGenerator(ImageGenerator):
    """A subclass of ImageGenerator specifically designed to use the Shuttle 3 Diffusion
    model for faster image generation.

    Attributes:
        pipe (DiffusionPipeline): The Shuttle 3 Diffusion model for image generation.

    Methods:
        _init_gen_model(): Initializes the Shuttle 3 Diffusion model.
        generate_images_batch(prompts, negative_prompt, prompt_objects): Generates a batch of images based on the provided prompts.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the Shuttle3DiffusionImageGenerator with the given arguments."""
        super().__init__(*args, **kwargs)
        self.pipe = self._init_gen_model()

    def _init_gen_model(self):
        """Initializes the Shuttle 3 Diffusion model for image generation.

        Returns:
            DiffusionPipeline: The initialized Shuttle 3 Diffusion model.
        """
        ckpt = "shuttleai/shuttle-3-diffusion"

        if self.device == "cpu":
            logger.info("Loading Shuttle 3 Diffusion on CPU...")
            pipe = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
            pipe.to("cpu")
        else:
            logger.info("Loading Shuttle 3 Diffusion on GPU...")
            pipe = DiffusionPipeline.from_pretrained(
                ckpt,
                torch_dtype=torch.bfloat16,
            )

            gpu_id = torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(gpu_id).total_memory / (
                1024**3
            )
            if total_vram < 25:
                logger.info("Quantizing the model because VRAM is less than 25 GB...")

                quantize(pipe.transformer, weights=qint8)
                freeze(pipe.transformer)
                quantize(pipe.text_encoder, weights=qint8)
                freeze(pipe.text_encoder)
                quantize(pipe.text_encoder_2, weights=qint8)
                freeze(pipe.text_encoder_2)
                quantize(pipe.vae, weights=qint8)
                freeze(pipe.vae)

            pipe.to("cuda")
            if total_vram < 21:
                logger.info(
                    "Enabling model CPU offload the total VRAM is less than 21 GB..."
                )
                pipe.enable_model_cpu_offload()

        return pipe

    def generate_images_batch(
        self,
        prompts: List[str],
        negative_prompt: str,
        prompt_objects: Optional[List[List[str]]] = None,
    ) -> List[Image.Image]:
        """Generates a batch of images using the Shuttle 3 Diffusion model based on the
        provided prompts.

        Args:
            prompts (List[str]): A list of positive prompts to guide image generation.
            negative_prompt (str): The negative prompt to avoid certain features in the image.
            prompt_objects (Optional[List[List[str]]]): Optional list of objects for each prompt for CLIP model testing.

        Returns:
            List[Image.Image]: A list of generated images.
        """

        image = self.pipe(
            prompts,
            guidance_scale=3.5,
            num_inference_steps=4,
            max_sequence_length=256,
            height=512,
            width=512
            # generator=torch.Generator("cpu").manual_seed(0)
        ).images

        return image

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache."""
        self.pipe.reset_device_map()
        self.pipe = self.pipe.to("cpu")
        if self.use_clip_image_tester:
            self.clip_image_tester.release()
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import os

    # Create the generator
    image_generator = Shuttle3DiffusionImageGenerator(
        seed=42,
        use_clip_image_tester=False,
        image_tester_patience=1,
        batch_size=4,
        device="cuda",
    )
    prompts = [
        "A photo of a bicycle pedaling alongside an aeroplane.",
        # "A photo of a dragonfly flying in the sky.",
        "A photo of a dog walking in the park.",
        "A photo of an alien exploring the galaxy.",
        "A photo of a robot working on a computer.",
    ] * 128

    image_paths = []
    counter = 0
    for generated_images_batch in image_generator.generate_images(prompts):
        for generated_image in generated_images_batch:
            image_path = os.path.join("./", f"shuttle_3_{counter}.jpg")
            generated_image.save(image_path)
            image_paths.append(image_path)
            counter += 1

    image_generator.release(empty_cuda_cache=True)
