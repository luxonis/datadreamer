from PIL import Image
import torch
from diffusers import DiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType
from typing import List, Optional

from basereduce.image_generation.image_generator import ImageGenerator


class StableDiffusionImageGenerator(ImageGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base, self.refiner = self._init_gen_model()
        self.base_processor, self.refiner_processor = self._init_processor()

    def _init_gen_model(self):
        # Load the model and processor here
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
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

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_objects: Optional[List[str]] = None,
    ) -> Image.Image:
        if prompt_objects is not None:
            for obj in prompt_objects:
                prompt = prompt.replace(obj, f"({obj})1.5", 1)

        print(prompt)

        conditioning, pooled = self.base_processor(prompt)
        conditioning_neg, pooled_neg = self.base_processor(negative_prompt)

        conditioning_refiner, pooled_refiner = self.refiner_processor(prompt)
        negative_conditioning_refiner, negative_pooled_refiner = self.refiner_processor(
            negative_prompt
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

    def release(self, empty_cuda_cache=False) -> None:
        self.base = self.base.to("cpu")
        self.refiner = self.refiner.to("cpu")
        if self.use_clip_image_tester:
            self.clip_image_tester.release()
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()
