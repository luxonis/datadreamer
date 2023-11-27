from abc import ABC, abstractmethod
from typing import Optional, Union, List
import enum
import os
import json
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType
from tqdm import tqdm
import random


# Enum for generative model names
class GenModelName(enum.Enum):
    STABLE_DIFFUSION_XL = "stabilityai/stable-diffusion-xl-base-1.0"
    # Add more models as needed


# Abstract base class for image generation
class ImageGenerator(ABC):
    def __init__(
        self,
        model_name: GenModelName,
        prompt_prefix: Optional[str] = None,
        prompt_suffix: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[float] = None,
    ) -> None:
        self.seed = seed
        self.model_name = model_name
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.negative_prompt = negative_prompt
        if seed is not None:
            self.set_seed(seed)

    @abstractmethod
    def generate_images(self):
        pass

    @abstractmethod
    def _test_image(self):
        pass

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class StableDiffusionImageGenerator(ImageGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base, self.refiner = self._init_gen_model()
        self.base_processor, self.refiner_processor = self._init_processor()

    def _init_gen_model(self):
        # Load the model and processor here
        base = DiffusionPipeline.from_pretrained(
            self.model_name.value,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        base.enable_model_cpu_offload()
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner.enable_model_cpu_offload()

        return base, refiner

    def _init_processor(self):
        compel = Compel(
            tokenizer=[self.base.tokenizer, self.base.tokenizer_2],
            text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        compel_refiner = Compel(
            tokenizer=[self.refiner.tokenizer_2],
            text_encoder=[self.refiner.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[True],
        )
        return compel, compel_refiner

    def generate_images(self, prompts: List[str]) -> List[Image.Image]:
        images = []
        for prompt in tqdm(prompts):
            images.append(self._generate_image(prompt))
        return images

    def _generate_image(self, prompt: str) -> Image.Image:
        prompt = self.prompt_prefix + prompt + self.prompt_suffix
        conditioning, pooled = self.base_processor(prompt)
        conditioning_neg, pooled_neg = self.base_processor(self.negative_prompt)

        conditioning_refiner, pooled_refiner = self.refiner_processor(prompt)
        negative_conditioning_refiner, negative_pooled_refiner = self.refiner_processor(
            self.negative_prompt
        )
        image = self.base(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=conditioning_neg,
            negative_pooled_prompt_embeds=pooled_neg,
            num_inference_steps=65,
            denoising_end=0.78,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt_embeds=conditioning_refiner,
            pooled_prompt_embeds=pooled_refiner,
            negative_prompt_embeds=negative_conditioning_refiner,
            negative_pooled_prompt_embeds=negative_pooled_refiner,
            num_inference_steps=65,
            denoising_start=0.78,
            image=image,
        ).images[0]

        return image

    def _test_image(self, image: Image.Image) -> bool:
        # Implement the image testing logic here
        # For now, we return True to indicate a valid image
        return True
