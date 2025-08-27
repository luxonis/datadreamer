from __future__ import annotations

from typing import List, Optional

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image
from safetensors.torch import load_file

from datadreamer.image_generation.image_generator import ImageGenerator


class StableDiffusionLightningImageGenerator(ImageGenerator):
    """A subclass of ImageGenerator specifically designed to use the Stable Diffusion
    Lightning model for faster image generation.

    Attributes:
        pipe (StableDiffusionXLPipeline): The Stable Diffusion Lightning model for image generation.

    Methods:
        _init_gen_model(): Initializes the Stable Diffusion Lightning model.
        _init_compel(): Initializes the Compel model for text prompt weighting.
        generate_images_batch(prompts, negative_prompt, prompt_objects): Generates a batch of images based on the provided prompts.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the StableDiffusionLightningImageGenerator with the given
        arguments."""
        super().__init__(*args, **kwargs)
        self.pipe = self._init_gen_model()
        self.compel = self._init_compel()

    def _init_gen_model(self) -> StableDiffusionXLPipeline:
        """Initializes the Stable Diffusion Lightning model for image generation.

        Returns:
            StableDiffusionXLPipeline: The initialized Stable Diffusion Lightning model.
        """
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!
        config = UNet2DConditionModel.load_config(base, subfolder="unet")

        logger.info(f"Initializing SDXL Lightning on {self.device}...")
        if self.device == "cpu":
            unet = UNet2DConditionModel.from_config(config)
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
            pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet)
        else:
            unet = UNet2DConditionModel.from_config(config).to(
                self.device, torch.float16
            )
            unet.load_state_dict(
                load_file(hf_hub_download(repo, ckpt), device=self.device)
            )
            pipe = StableDiffusionXLPipeline.from_pretrained(
                base, unet=unet, torch_dtype=torch.float16, variant="fp16"
            ).to(self.device)
            pipe.enable_model_cpu_offload()

        # Ensure sampler uses "trailing" timesteps.
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        return pipe

    def _init_compel(self) -> Compel:
        """Initializes the Compel model for text prompt weighting.

        Returns:
            Compel: The initialized Compel model.
        """
        compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.device,
        )
        return compel

    def generate_images_batch(
        self,
        prompts: List[str],
        negative_prompt: str,
        prompt_objects: Optional[List[List[str]]] = None,
    ) -> List[Image.Image]:
        """Generates a batch of images using the Stable Diffusion Lightning model based
        on the provided prompts.

        Args:
            prompts (List[str]): A list of positive prompts to guide image generation.
            negative_prompt (str): The negative prompt to avoid certain features in the image.
            prompt_objects (Optional[List[List[str]]]): Optional list of objects for each prompt for CLIP model testing.

        Returns:
            List[Image.Image]: A list of generated images.
        """

        if prompt_objects is not None:
            for i in range(len(prompt_objects)):
                for obj in prompt_objects[i]:
                    prompts[i] = prompts[i].replace(obj, f"({obj})1.5", 1)

        conditioning, pooled = self.compel(prompts)
        conditioning_neg, pooled_neg = self.compel([negative_prompt] * len(prompts))
        images = self.pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=conditioning_neg,
            negative_pooled_prompt_embeds=pooled_neg,
            guidance_scale=0.0,
            num_inference_steps=4,
        ).images

        return images

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache."""
        self.pipe = self.pipe.to("cpu")
        if self.use_clip_image_tester:
            self.clip_image_tester.release()
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import os

    # Create the generator
    image_generator = StableDiffusionLightningImageGenerator(
        seed=42,
        use_clip_image_tester=False,
        image_tester_patience=1,
        batch_size=4,
        device="cuda",
    )
    prompts = [
        "A photo of a bicycle pedaling alongside an aeroplane.",
        "A photo of a dragonfly flying in the sky.",
        "A photo of a dog walking in the park.",
        "A photo of an alien exploring the galaxy.",
        "A photo of a robot working on a computer.",
    ]
    prompt_objects = [
        ["aeroplane", "bicycle"],
        ["dragonfly"],
        ["dog"],
        ["alien"],
        ["robot", "computer"],
    ]

    image_paths = []
    counter = 0
    for generated_images_batch in image_generator.generate_images(
        prompts, prompt_objects
    ):
        for generated_image in generated_images_batch:
            image_path = os.path.join("./", f"image_lightning_{counter}_no_compel.jpg")
            generated_image.save(image_path)
            image_paths.append(image_path)
            counter += 1

    image_generator.release(empty_cuda_cache=True)
