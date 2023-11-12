from basereduce.prompt_generation import LMPromptGenerator, LMName


if __name__ == "__main__":
    object_names = ['aeroplane', 'bicycle', 'bird', 'boat']
    prompt_generator = LMPromptGenerator(class_names=object_names, model_name=LMName.MISTRAL)
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)