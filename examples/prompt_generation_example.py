from datadreamer.prompt_generation import LMPromptGenerator

if __name__ == "__main__":
    object_names = ["aeroplane", "bicycle", "bird", "boat"]
    prompt_generator = LMPromptGenerator(class_names=object_names)
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)
