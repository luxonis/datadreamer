from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image
from typing import List, Optional

# from basereduce.image_generation.generative_model import GenerativeModel
from basereduce.image_generation.image_generator import ImageGenerator


class StableDiffusionTurboImageGenerator(ImageGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = self._init_gen_model()

    def _init_gen_model(self):
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
        print(prompt)

        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
        ).images[0]

        return image

    def release(self, empty_cuda_cache=False) -> None:
        self.base = self.base.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()
