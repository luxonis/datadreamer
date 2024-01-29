import random
import re
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from datadreamer.prompt_generation.lm_prompt_generator import LMPromptGenerator


class TinyLlamaLMPromptGenerator(LMPromptGenerator):
    """A language model-based prompt generator class, extending PromptGenerator.

    Attributes:
        device (str): Device to run the language model on ('cuda' for GPU, 'cpu' for CPU).
        model (AutoModelForCausalLM): The pre-trained causal language model for generating prompts.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.

    Methods:
        _init_lang_model(): Initializes the language model and tokenizer.
        generate_prompts(): Generates a list of prompts based on the class names.
        _create_lm_prompt_text(selected_objects): Creates a text prompt for the language model.
        generate_prompt(prompt_text): Generates a single prompt using the language model.
        _test_prompt(prompt, selected_objects): Tests if the generated prompt is valid.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(
        self,
        class_names: List[str],
        prompts_number: int = 10,
        num_objects_range: Optional[List[int]] = None,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        """Initializes the LMPromptGenerator with class names and other settings."""
        super().__init__(class_names, prompts_number, num_objects_range, seed, device)

    def _init_lang_model(self):
        """Initializes the language model and tokenizer for prompt generation.

        Returns:
            tuple: The initialized language model and tokenizer.
        """
        if self.device == "cpu":
            print("Loading language model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
        else:
            print("Loading language model on GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)
        print("Done!")
        return model.to(self.device), tokenizer
    

    # def generate_prompts(self) -> List[str]:
    #     """Generates a list of text prompts based on the class names.

    #     Returns:
    #         List[str]: A list of generated prompts.
    #     """
    #     prompts = []
    #     for _ in tqdm(range(self.prompts_number), desc="Generating prompts"):
    #         selected_objects = random.sample(
    #             self.class_names, random.randint(*self.num_objects_range)
    #         )
    #         prompt_text = self._create_lm_prompt_text(selected_objects)
    #         generated_prompt = self.generate_prompt(prompt_text)
    #         prompts.append((selected_objects, generated_prompt))
    #     return prompts
    
    # def generate_prompts(self) -> List[str]:
    #     """Generates a list of text prompts based on the class names.

    #     Returns:
    #         List[str]: A list of generated prompts.
    #     """
    #     prompts = []
    #     for i in tqdm(range(self.prompts_number), desc="Generating prompts"):
    #         selected_objects = random.sample(
    #             self.class_names, random.randint(*self.num_objects_range)
    #         )
    #         prompt_text = self._create_lm_prompt_text(selected_objects)
    #         correct_prompt_generated = False
    #         number_of_tries = 0
    #         while not correct_prompt_generated and number_of_tries < 5:
    #             generated_prompt = self.generate_prompt(prompt_text)
    #             number_of_tries += 1
    #             print(f"{i+1}. image caption generated on attempt nr. {number_of_tries}: {generated_prompt}")
    #             if self._test_prompt(generated_prompt, selected_objects):
    #                 prompts.append((selected_objects, generated_prompt))
    #                 correct_prompt_generated = True
    #                 print("Success!\n")
    #     return prompts

    def remove_incomplete_sentence(self, text):
        # Define the regex pattern to capture up to the last sentence-ending punctuation
        pattern = r'^(.*[.!?;:])'
        match = re.search(pattern, text)
        return match.group(0) if match else text

    def _create_lm_prompt_text(self, selected_objects: List[str]) -> str:
        """Creates a language model text prompt based on selected objects.

        Args:
            selected_objects (List[str]): Objects to include in the prompt.

        Returns:
            str: A text prompt for the language model.
        """
        # return f"<|system|>\nYou are a chatbot who can help write prompts!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description.</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for image generation!</s>\n<|user|>\nGenerate a short and concise caption for an image. The objects in the image are: {', '.join(selected_objects)}. The objects interact in a meaningful way within the scene. The caption contains a short scene description. The captions starts with the words 'A photo of '!</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: 'A photo of '!</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. The caption must follow this template: 'A photo of {', '.join(selected_objects)}'! The objects interact in a meaningful way within a scene, complete with a very short scene description.</s>\n<|assistant|>\n" # The caption must start with the words 'A photo of '!
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. The caption must follow this template: 'A photo of {', '.join(selected_objects)}'! The objects interact in a meaningful way within the scene that is very shortly described. The caption must start with the words 'A photo of '!</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. The caption must follow this template: 'A photo of {', '.join(selected_objects)}'! The objects interact in a meaningful way within the scene that is very shortly described. </s>\n<|assistant|>\nA photo of "
        # return f"<|system|>\nYou are a chatbot who helps write caption for an image!</s>\n<|user|>\nGenerate a short and concise caption for an image. The image contains these objects: {', '.join(selected_objects)}. The objects interact in a meaningful way within the scene that is very shortly described. The caption must start with the words: 'A photo of '!</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption starts with the words 'A photo of '!</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene. The caption must start with the words: 'A photo of '!</s>\n<|assistant|>\n"
        # return f"Generate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene. The caption must start with the words: 'A photo of '!\nOutput: "
        # return f"Generate a short and concise caption for an image. You must follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene. Do not write any emojis. The caption must be short and start with the words: 'A photo of '!\nOutput: "
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: 'A photo of '!\nHere are 2 examples:\n- 'aeroplane', 'boat', 'bicycle': 'A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.'\n- 'bicycle': 'A photo of bicycles along a scenic mountain path, where the riders seem to have taken a moment to appreciate the stunning views.'\nGenerate such a caption for these objects: {', '.join(selected_objects)}!</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGiven this example:\n- 'aeroplane', 'boat', 'bicycle': 'A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.'\n\nGenerate a short and concise caption for an image containing these objects: {', '.join(selected_objects)}. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: 'A photo of '!\n</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: 'A photo of '!\n</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: 'A photo of '!\n</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: 'A photo of '! Do not write something like 'Caption reads'.</s>\n<|assistant|>\n"
        # return f"<|system|>\nYou are a chatbot who helps write captions for images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: 'A photo of '! Do not write phrase 'Caption reads'.</s>\n<|assistant|>\n"
        return f"<|system|>\nYou are a chatbot who describes content of images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: 'A photo of '! Do not use the phrase 'Caption reads'.</s>\n<|assistant|>\n"
    

    def generate_prompt(self, prompt_text: str) -> str:
        """Generates a single prompt using the language model.

        Args:
            prompt_text (str): The text prompt for the language model.

        Returns:
            str: The generated prompt.
        """
        encoded_input = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **encoded_input,
            # max_new_tokens=100,
            max_new_tokens=70,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_beams=1,
            # num_beams=3,
            # do_sample=False,
            # num_beams=5,
            # do_sample=True,
            # top_p=0.95,
            # top_k=30,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded_prompt = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        # print(decoded_prompt)
        # instructional_pattern = r"<\|system\|>\n.*?\n<\|user\|>\n.*?\n<\|assistant\|>\n"
        # instructional_pattern = r"<\|system\|>\n.*?<\/s>\n<\|user\|>\n.*?\n<\/s>\n<\|assistant\|>\n"
        instructional_pattern = r"<\|system\|>\n.*?\n<\|user\|>\n.*?\n<\|assistant\|>\n"
        # Remove the instructional text to isolate the caption
        decoded_prompt = (
            re.sub(instructional_pattern, "", decoded_prompt)
            .replace('"', "")
            .replace("'", "")
        )

        return self.remove_incomplete_sentence(decoded_prompt)


if __name__ == "__main__":
    # Example usage of the class
    object_names = ["aeroplane", "bicycle", "bird", "boat", "city"]
    prompt_generator = TinyLlamaLMPromptGenerator(
        class_names=object_names, prompts_number=5, device="cpu"
    )
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)
    print("="*100)
    print("New objects!")
    print("="*100)
    object_names = ["city", "cat", "dog", "horse"]
    prompt_generator = TinyLlamaLMPromptGenerator(
        class_names=object_names, prompts_number=5, device="cpu"
    )
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)
# Loading language model on CPU...
# Done!
# Generating prompts:   0%|                                                                                                                                                                               | 0/2 [00:00<?, ?it/s]Selected objects: ['aeroplane', 'boat', 'bicycle']
# Prompt text: <|system|>
# You are a chatbot who can help write prompts!</s>
# <|user|>
# Generate a short and concise caption for an image. Follow this template: 'A photo of aeroplane, boat, bicycle', where the objects interact in a meaningful way within a scene, complete with a short scene description.</s>
# <|assistant|>

# Generating prompts:  50%|███████████████████████████████████████████████████████████████████████████████████▌                                                                                   | 1/2 [00:28<00:28, 28.50s/it]Selected objects: ['bicycle']
# Prompt text: <|system|>
# You are a chatbot who can help write prompts!</s>
# <|user|>
# Generate a short and concise caption for an image. Follow this template: 'A photo of bicycle', where the objects interact in a meaningful way within a scene, complete with a short scene description.</s>
# <|assistant|>

# Generating prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:53<00:00, 26.91s/it]
# (['aeroplane', 'boat', 'bicycle'], '<|system|>\nYou are a chatbot who can help write prompts! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of aeroplane, boat, bicycle, where the objects interact in a meaningful way within a scene, complete with a short scene description. \n<|assistant|>\nPhotograph of a Cessna 172, a kayak, and a bike ride, showcasing a vital role the aeroplane, kayak, and bike play in a memorable day trip filled with joy, freedom, and friendship. The caption reads: Celebrate the day')
# (['bicycle'], '<|system|>\nYou are a chatbot who can help write prompts! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of bicycle, where the objects interact in a meaningful way within a scene, complete with a short scene description. \n<|assistant|>\nA striking scene of a biker and his bicycle, captivating the viewer with a vibrant blend of colors and textures. The bicycle engages in playful exchanges with the biker, inviting contemplation of their role in a meaningful moment. The caption reads, Life on')

# TODO: 
# Doesn't work the removing template part - 
# Doesn't follow the template
# (['aeroplane', 'boat', 'bicycle'], 'Photograph of a Cessna 172, a kayak, and a bike ride, showcasing a vital role the aeroplane, kayak, and bike play in a memorable day trip filled with joy, freedom, and friendship. The caption reads: Celebrate the day')
# (['bicycle'], 'A striking scene of a biker and his bicycle, captivating the viewer with a vibrant blend of colors and textures. The bicycle engages in playful exchanges with the biker, inviting contemplation of their role in a meaningful moment. The caption reads, Life on')
        
# Prompt V2
# (['aeroplane', 'boat', 'bicycle'], 'Snap away! A majestic aeroplane soars over a serene lake, while a charming bicycle sparsens its way down a winding road. \nThe aeroplane and bicycle interact in this picturesque, serene scene, serving as the visual and auditory backdrop to the lush')
# (['bicycle'], 'A charming scene as biker and his bicycle share the bike lane together, creating a dynamic moment between two iconic symbols of freedom and travel.\n\n(Starts with A photo of...)')
        
# Prompt V3
# (['aeroplane', 'boat', 'bicycle'], 'Photograph of a Canoe and Infrastructure, a Busy Scene\n\nIn the heart of town, theres nothing but cars and infrastructure, yet the canoe on the nearby canal catches the eye. The picturesque, yet busy city is waiting to be explored, with the bustling skyline')
# (['bicycle'], 'A photo of a BICYCLE, a bike making a triumphant pass between fading olive trees and lush green pastures. The suns warm rays cast glints between the vines, warming the bikers torsos and creating a peaceful harmony on the quintessential gra')
# (env) 

# Prompt V4
# (['aeroplane', 'boat', 'bicycle'], 'Photograph of a Flightless Bird, Ferry, and Bike\n\n🚴\u200d♂️ Traveler: Wow, what an amazing view! Those aeroplanes flying overhead, the brightly colored boats, and that sweet ride going by on a bike - Im in heaven!')
# (['bicycle'], 'A photo of a group biking together, their movements entwining around the pavement. A few pedaling away, others taking a brief break to adjust their helmets while keeping vigilant on their surroundings.')
        
# Prompt V5
# (['aeroplane', 'boat', 'bicycle'], 'A boat and a bicycle in a calm sea, just like the sight of a bird flirting with the sun!')
# (['bicycle'], 'Meet the perfect mode of transportation for the city on two wheels! Step aboard this picturesque, bike-friendly street with joy and freedom.')
# (['aeroplane'], 'Without a doubt, heres an exciting and thrilling photo of the majestic aeroplane! This shot captures the exhilarating feeling of soaring through the skies, complete with a glittering sunset and the sound of the engine roaring. Pulse-pounding adrenaline pumps through your veins as you take in the vibr')
# (['aeroplane', 'bird', 'bicycle'], 'A Picture of a Bird, Bicycle, and Aeroplane\nA vibrant image of a world full of fascinating things. The picture is a perfect blend of nature and art. In this scenery, the bird, bicycle, and aeroplane each play a vital role in creating a sense of harmony.')

# Prompt V6
# (['aeroplane', 'boat', 'bicycle'], 'A photo of 🚴\u200d♀️, 🚵\u200d♂️, and a boat in the ocean, with the words Surf the waves and coast to shore below the image on background. The shot is visually stunning, capturing the freedom and thrill of the outdoors.')
# (['bicycle'], 'A photo of 🚴\u200d♀️, with the biker posing confidently, wearing a vibrant motorcycle jacket and leather pants. The scenery is an intimate town square, with locals chatting and enjoying a picnic. The background is a quaint old street lined with colorful street signs and old wooden benches. As the sun sets, the bikers gather around the group,')
# (['aeroplane'], 'A photo of 🛳️ a plane taking off.')
# (['aeroplane', 'bird', 'bicycle'], 'A photo of 🚜, 🚹, and 🚱 within a scene that conveys the power and freedom of flight, nature, and biking.\n\n🚜: Aeroplane taking off in the sky\n🚹: Bicyclist riding a pensive route\n🚱: Bird flying gracefully above the canopy of trees')

# Prompt V7
# (['aeroplane', 'boat', 'bicycle'], 'Snap away! A plane soars over a sunset, biking is a blast, and two boats peacefully sway on the water. Witness the magic of the moment on this stunning shot!')
# (['bicycle'], 'Meet the riders of the city on two wheels! - Bicycle Conversation 101')
# (['aeroplane'], 'A breathtaking view of the majestic skyline, lit up by the fading twilight (Aeroplane flies overhead)\n\nThe image perfectly captures the majesty of the skyline against a warm tinted background. The camera captures the aeroplane, which seems to float at the horizon, and the setting sun provides a peaceful backdrop.\nI hope you enjoy this image')
# (['aeroplane', 'bird', 'bicycle'], 'In an amazing world of aerial sightseeing, you see this plane and its amazing speed and direction. In a scenic town with bikes around, you spot this bicycle going through the streets with ease! The scene is full of intriguing things which made your heart skip a beat. This short and concise caption is a perfect way to showcase the breathtaking images capturing the essence of')

# Prompt V8
# (['aeroplane', 'boat', 'bicycle'], 'Photograph of [image description] with [object interacting with other objects]\nRiding the rails: A train passes us by, the aeroplane bobs in the water, and the bicycle circles around the scene. Captures the essence of this wonderful scenic view.')
# (['bicycle'], 'A picture of a bicycle, a peaceful scene in a lush landscape\n\nAs the bike lurches through the meandering path lined with rolling leaves, the suns warm rays cast glints between the trees, evoking a serene calm')
# (['aeroplane'], 'A picture of a plane flying over a cityscape, with birds, trees, and a glistening sky in the background. The caption A photo of an aeroplane, transporting passengers to amazing destinations!')
# (['aeroplane', 'bird', 'bicycle'], 'Amidst the tranquil fields and the rustling leaves, a bird in flight perches up with an aeroplane gliding down below, merging seamlessly into the bluish sky. With a bicycle cycling on a straight path, its a joyous sight to behold, creating a vibrant cycle of action and reflection. The perfect scene for contemplation. A photo of an ')

# Test prompt V9 - sampling
# Test prompt V9_2 - sampling
# Test prompt V9_3 - beam search

# Test prompt V10
# (['aeroplane', 'city', 'bird'], 'Generate a short and concise caption for an image. Follow this template: A photo of aeroplane, city, bird, where the objects interact in a meaningful way within a scene. The caption must start with the words: A photo of !\nOutput: 👉 A photo of a bird taking flight, a mountain in the background!')
# (['bicycle', 'city'], 'Generate a short and concise caption for an image. Follow this template: A photo of bicycle, city, where the objects interact in a meaningful way within a scene. The caption must start with the words: A photo of !\nOutput:  A Photo of Bicycle, City!')
# (['aeroplane'], 'Generate a short and concise caption for an image. Follow this template: A photo of aeroplane, where the objects interact in a meaningful way within a scene. The caption must start with the words: A photo of !\nOutput:  A photo of an aeroplane at sea level hovering above the earth. The plane rotates, indicating that it is going above the sky and below the ocean. The image presents a simple yet interesting scenario as this aeroplane can be seen hovering, with the sea and the sky providing a natural frame for the scene.')
# (['city', 'aeroplane', 'bird'], 'Generate a short and concise caption for an image. Follow this template: A photo of city, aeroplane, bird, where the objects interact in a meaningful way within a scene. The caption must start with the words: A photo of !\nOutput:  A photo of a city, aeroplane, bird - symbolizing growth, progress, and achievement')
# (['aeroplane', 'city'], 'Generate a short and concise caption for an image. Follow this template: A photo of aeroplane, city, where the objects interact in a meaningful way within a scene. The caption must start with the words: A photo of !\nOutput: 3D aeroplane flying over bustling city street. 5. Title Caption for the Image:\na. Write a compelling and engaging title that captures the essence of the image. b. Make sure that the title accurately predicts the visual content displayed in the image. For example: Striking')
        
# TODO:
# Put into the prompt an example!!! See if that helps
# (['aeroplane', 'city', 'bird'], 'Generate a short and concise caption for an image. You must follow this template: A photo of aeroplane, city, bird, where the objects interact in a meaningful way within a scene. Do not write any emojis. The caption must be short and start with the words: A photo of !\nOutput: 👉 [image_path] In #NYC with [object_name]! Photo taken on [date_stamp]. Enjoy your flight! Share the love on social media by using the hashtags: #NYCtrips & #AmazingNYC.')
# (['bicycle', 'city'], 'Generate a short and concise caption for an image. You must follow this template: A photo of bicycle, city, where the objects interact in a meaningful way within a scene. Do not write any emojis. The caption must be short and start with the words: A photo of !\nOutput: 🚣\u200d♀️🛶 A photo of a woman jumping off a bike and jumping into a fountain in a city square.')
# (['aeroplane'], 'Generate a short and concise caption for an image. You must follow this template: A photo of aeroplane, where the objects interact in a meaningful way within a scene. Do not write any emojis. The caption must be short and start with the words: A photo of !\nOutput: \nA photo of an aeroplane asserting ownership over a field of wheat.')
# (['city', 'aeroplane', 'bird'], 'Generate a short and concise caption for an image. You must follow this template: A photo of city, aeroplane, bird, where the objects interact in a meaningful way within a scene. Do not write any emojis. The caption must be short and start with the words: A photo of !\nOutput:  Nature: a bird perched comfortably on a branch, staring out at the world below.\nA photo of city, bicycle, bus, helicopter, lamp post, skyscraper, train, bridge, street lamp, windmill, truck, car, van, plane in a green field.\n')
# (['aeroplane', 'city'], 'Generate a short and concise caption for an image. You must follow this template: A photo of aeroplane, city, where the objects interact in a meaningful way within a scene. Do not write any emojis. The caption must be short and start with the words: A photo of !\nOutput:  A photo of aeroplane, city')

# (['aeroplane', 'boat', 'bicycle'], 'A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.')
# (['bicycle'], 'A photo of bicycles along a scenic mountain path, where the riders seem to have taken a moment to appreciate the stunning views.')
# - 'aeroplane', 'boat', 'bicycle': 'A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.'\n- 'bicycle': 'A photo of bicycles along a scenic mountain path, where the riders seem to have taken a moment to appreciate the stunning views.'
        
# Prompt V11 - 2shot
# (['aeroplane', 'city', 'bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of aeroplane, city, bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\nHere are 2 examples:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n- bicycle: A photo of bicycles along a scenic mountain path, where the riders seem to have taken a moment to appreciate the stunning views.\nGenerate such a caption for these objects: aeroplane, city, bird! \n<|assistant|>\nPhotos of Aeroplane: In a cozy scene with birds,\nthe sky transforms into a painted canvas bright.\nThe aeroplane flies above the city,\nlike a symbol of human ingenuity and strength.\nThe bird in flight is a serene tribute,\nbringing joy and adm')
# (['bicycle', 'city'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of bicycle, city, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\nHere are 2 examples:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n- bicycle: A photo of bicycles along a scenic mountain path, where the riders seem to have taken a moment to appreciate the stunning views.\nGenerate such a caption for these objects: bicycle, city! \n<|assistant|>\nA photo of Bicycle, City!\nA moment of harmony between human-powered and mechanical transportation. The bicyclists eyes show their satisfaction in sharing the road, while the vehicles in the background seem to offer a sense of peace and connection. The citys greenery and buildings are beautifully')
# (['aeroplane'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of aeroplane, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\nHere are 2 examples:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n- bicycle: A photo of bicycles along a scenic mountain path, where the riders seem to have taken a moment to appreciate the stunning views.\nGenerate such a caption for these objects: aeroplane! \n<|assistant|>\n💨 A photo of aeroplane 💨\n\nIn this stunning image showcasing the symbiotic relationship between human-powered and mechanized transportation, boisterous bicyclists pedal alongside an aeroplane taking off. The aeroplanes powerful engine fills the frame,')
# (['city', 'aeroplane', 'bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of city, aeroplane, bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\nHere are 2 examples:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n- bicycle: A photo of bicycles along a scenic mountain path, where the riders seem to have taken a moment to appreciate the stunning views.\nGenerate such a caption for these objects: city, aeroplane, bird! \n<|assistant|>\nA photo of city and aeroplane, seamlessly arranged within a beautiful composition that showcases the interconnectivity of human and mechanical transportation. The bicycles on the route give way to the aeroplane, taking us above the serene landscape, while the birds peck at the grass around us, creating an all-en')
# (['aeroplane', 'city'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of aeroplane, city, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\nHere are 2 examples:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n- bicycle: A photo of bicycles along a scenic mountain path, where the riders seem to have taken a moment to appreciate the stunning views.\nGenerate such a caption for these objects: aeroplane, city! \n<|assistant|>\nA photo of Aeroplane & City\n\nA photo where the aeroplane blends effortlessly into the urban landscape, surrounded by the sparkling cityscape and skylines. Cars zoom past as the passengers take a moment to appreciate the beauty of this city as it unfolds around them. The chaotic commotion')

# Prompt V12 - 1shot
# (['aeroplane', 'city', 'bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGiven this example:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n\nGenerate a short and concise caption for an image containing these objects: aeroplane, city, bird. Follow this template: A photo of aeroplane, city, bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA breathtaking view of a bird taking flight alongside an aeroplane, a feat that has been marred by centuries of innovation and technological advancements!\n\nThis shot captures the perfect symphony between human-powered and mechanical transportation, as the winged insect soars high above the clouds,')
# (['bicycle', 'city'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGiven this example:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n\nGenerate a short and concise caption for an image containing these objects: bicycle, city. Follow this template: A photo of bicycle, city, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA photo of an aeroplane taking off, alongside a bicycle that pedals alongside it, showcasing the harmony between human-powered and mechanical transportation. The image captures the majesty of the city skyline, complete with its iconic landmarks.')
# (['aeroplane'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGiven this example:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n\nGenerate a short and concise caption for an image containing these objects: aeroplane. Follow this template: A photo of aeroplane, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA striking comparison between two human-powered vehicles - a bicycle pedaling alongside an aeroplane taking off, showing the harmony between those two modes of transport.')
# (['city', 'aeroplane', 'bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGiven this example:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n\nGenerate a short and concise caption for an image containing these objects: city, aeroplane, bird. Follow this template: A photo of city, aeroplane, bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nAirborne sightseeing along the coastline with a sleek aeroplane gliding in the direction of the city. - Vivid cityscape with birds darting among skyscrapers while aerial vehicles move with grace through the sky.')
# (['aeroplane', 'city'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGiven this example:\n- aeroplane, boat, bicycle: A photo of a bicycle pedaling alongside an aeroplane taking off, showcasing the harmony between human-powered and mechanical transportation.\n\nGenerate a short and concise caption for an image containing these objects: aeroplane, city. Follow this template: A photo of aeroplane, city, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA photo of an aeroplane taking off, featuring a bustling city in the background! Watch as the mechanical transportation seamlessly interacts with the natural surroundings, adding another dimension to the scene.')

# Prompt V13 + new gen params
# (['aeroplane', 'city', 'bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of aeroplane, city, bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA bird flies past a majestic aeroplane, its wings spread wide, creating a serene and peaceful scene. The pair of creatures glide effortlessly above the bustling city, their wings catching the light in a dazzling display of motion. The image captures the essence of a harmonious relationship')
# (['bicycle', 'city'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of bicycle, city, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA striking scene of a bicycle parked alongside a busy street, with the pavement caked with mud from a recent downpour. A passerby stops to admire the bike, their expression one of wonder and appreciation. The sun beats down on the city below, casting long shadows across the road, while a')
# (['aeroplane'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of aeroplane, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA photo of an aeroplane flying over a lush green landscape, with a gentle breeze blowing through the trees. The aircrafts wings sweep back and forth, while the propellers turn with a smooth, steady beat. The cameras view is from above, capturing the aircrafts sleek, futur')
# (['city', 'aeroplane', 'bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of city, aeroplane, bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA stunning aerial view of a bustling city, with a bird soaring gracefully above the towering skyscrapers. The citys bustling streets and towering buildings, perfectly captured in this stunning photo. With a short description, we can easily imagine ourselves standing in the middle of this metropolis,')
# (['aeroplane', 'city'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of aeroplane, city, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must start with the words: A photo of !\n \n<|assistant|>\nA photo of an aeroplane soaring over a bustling city! The plane and its cityscape are in perfect harmony, with the aircraft and buildings blending effortlessly into one another. The suns rays reflect off the windows, creating a striking silhouette against the sky. The caption reads: City')
        
# Prompt V14 + new gen params
# (['bicycle', 'horse', 'water'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of bicycle, horse, water, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA lively scene of a bicycle, horse, and a puddle! The bike and horse are interacting with the puddle, creating a dynamic and meaningful scene.\n\nA photo of the bicycle, horse, and puddle, complete with a brief scene description: A lively scene of a bicycle, horse, and a puddle, where the animals interact in a fun and engaging way.')
# (['bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA photo of a bird perched on a tree branch, their wings spread wide in a graceful display. The birds feathers shimmer in the sunlight, casting a warm, golden glow over the scene. The branch is still and serene, the air thick with the sound of a nearby birds song. Above them, the sky is a brilliant blue, punctuated by the occasional streak of a cloud. This image is a reminder of the')
# (['dog'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of dog, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA photo of a playful dog running through a lush green park, their tail wagging and their fur brushing against the leaves. The dogs paw is on the ground, ready for a quick run. The scene is set against a backdrop of tall trees, their leaves rustling in the breeze. A person stands near the dog, holding a toy in their hand, watching the two of them play. The caption reads, A photo of a')
# (['dog'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of dog, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA photo of a playful dog, the pup chasing after a ball, the playful tail wagging, and the owner laughing in the background. The scene is a happy family gathering, with the dog at the center of it all, enjoying the moment with his owners. The caption reads: A photo of a happy family, playing together, and enjoying each others company!')
# (['cat', 'bicycle', 'horse'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of cat, bicycle, horse, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA photo of a cute cat riding a bike, and a horse galloping alongside! In this charming scene, the animals interact in a playful and harmonious manner, making for a perfect photo opportunity.')
# (['bicycle', 'horse'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of bicycle, horse, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA stunning view of a bicycle and horse walking through a picturesque countryside, their movements intertwined in a harmonious symphony of motion. The scene is both serene and energizing, a testament to the enduring power of nature and the human spirit. A short description of the scene could be added, such as A peaceful oasis amidst lush green hills, this bustling city framed by the rolling hills and verdant')
# (['bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA photo of a bird soaring through the sky, with a flock of feathered friends following in their wake. The birds glide effortlessly, their wings beating in perfect unison, their calls echoing through the trees. This picture captures the essence of a peaceful day spent with nature, a perfect moment of harmony.')
# (['cat'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of cat, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA photo of a playful cat, perched on a couch with a stuffed toy, its fur brushing against the soft cushions. The image captures the warmth and love that this furry friend has for its human companion. The scene is complete with a brief description of the couch and toy, creating a sense of familiarity and comfort. The caption reads: A photo of a playful cat, content and cozy, with a stuffed')
# (['bicycle', 'cat', 'bird'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of bicycle, cat, bird, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA photo of a bicycle riding alongside a fluffy kitty, with a bird perched on a tree branch. This scene is a whimsical and playful combination of nature and animal-human interaction.')
# (['dog', 'horse', 'cat'], '<|system|>\nYou are a chatbot who helps write captions for images! \n<|user|>\nGenerate a short and concise caption for an image. Follow this template: A photo of dog, horse, cat, where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: A photo of !\n \n<|assistant|>\nA photo of a playful dog, a content cat, and a happy horse in a beautiful field, surrounded by lush greenery. This snapshot captures the essence of a happy day, with each animal embracing natures beauty.\n\nScene description: The sun shines down on a peaceful meadow, where a group of furry friends frolic and play. A gentle breeze whispers through the trees, and the birds chirp in harmony')

# Prompt V15 + new gen params (6m45s for 3 prompts)
# max_new_tokens=70,
# do_sample=True,
# top_p=0.95,
# top_k=50,
# temperature=0.7,
# num_beams=3, => slower
# (['bicycle', 'horse', 'water'], 'A photo of a bicycle, horse, and a water fountain, where the bicycle and horse interact with the water fountain in a meaningful way, creating a beautiful and serene scene. The caption reads: A photo of a bicycle, horse, and a tranquil water fountain')
# (['bird'], 'A photo of a bird perched on a branch, its wings outstretched as it surveys its surroundings. The birds feathers glisten in the sunlight, and its beak glints in the shadows. The scene is peaceful and serene, a reminder of the beauty and wonder of nature.')
# (['dog'], 'A photo of a playful puppy snuggling up to its owner, their fur-filled arms wrapped around each other in a cozy embrace. The scene is set in a cozy cottage, with a fire crackling in the fireplace and a warm blanket draped over the couch. The dogs')

# Prompt V16 + new gen params (2m20s for 3 prompts)
# max_new_tokens=70,
# do_sample=True,
# top_p=0.95,
# top_k=50,
# temperature=0.7,
# num_beams=1, => faster
# ['bicycle', 'horse', 'water'], 'A lively scene of a bicycle, horse, and a puddle! The bike and horse are interacting with the puddle, creating a dynamic and meaningful scene.\n\nCaption reads: Life is full of surprises, and this photo captures it all!')
# (['bird'], 'A photo of a bird, a feathered friend perched on a branch, its delicate wings gracefully flapping as it surveys the lush green landscape below. The image captures the essence of natures majesty, the bond between animal and environment, and the beauty of life.')
# (['dog'], 'A photo of a dog playing catch with a ball in a field of green grass, complete with a backdrop of trees in the background. The caption reads: Excitedly catching a foul ball, the dog and its furry friend play together in a lush landscape.')
        
# Prompt V17 + same args as V16 
# (['bicycle', 'horse', 'water'], 'A Photo of a Bicycle, Horse, and Water\n\nThe bicycle and horse ride alongside the water, each taking their turn in the center of the frame. The water gently glides and swirls around them, creating a serene and calming environment. This image captures the essence of a relaxing')
# (['bird'], 'A photo of a bird in flight, taking off from a tree branch and soaring high above the forest canopy. The birds wingspan extends far beyond the frame, and its movements are graceful and fluid. The leaves on the trees below rustle as the bird passes through them, adding to the overall sense of movement and energy.')
# (['dog'], 'A photo of a curious and playful dog sniffing and chasing a ball in a lively backyard scene.')
# (['dog'], 'A photo of a playful dog, their playful energy filling the frame as they chase after a ball in a lush garden. The scene is beautifully captured, the colors vibrant and the animals movements fluid and graceful.')
# (['cat', 'bicycle', 'horse'], 'A photo of a cute cat riding a bike, and a horse passing by in the background. A meaningful scene of two animals interacting in harmony!')

