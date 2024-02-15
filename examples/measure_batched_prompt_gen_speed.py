import time

from datadreamer.prompt_generation import LMPromptGenerator, TinyLlamaLMPromptGenerator

if __name__ == "__main__":
    time_per_prompt_dict = {"tinyllama": {}, "mistral_int4": {}, "mistral_fp16": {}}
    for prompt_generator_class, batch_sizes, generator_name in zip(
        [
            TinyLlamaLMPromptGenerator,
            lambda *args, **kwargs: LMPromptGenerator(
                *args, **kwargs, quantization="4bit"
            ),
            lambda *args, **kwargs: LMPromptGenerator(
                *args, **kwargs, quantization="none"
            ),
        ],
        [
            [2**i for i in range(0, 10)],  # tinyllama
            [2**i for i in range(0, 9)],  # mistral_int4
            [2**i for i in range(0, 8)],  # mistral_fp16
        ],
        time_per_prompt_dict.keys(),
    ):
        for batch_size in batch_sizes:
            prompt_generator = prompt_generator_class(
                class_names=["aeroplane", "bicycle", "bird"],
                prompts_number=1,
                batch_size=batch_size,
                device="cuda",
            )
            time_start = time.time()
            generated_prompts = prompt_generator.generate_prompts()
            print(f"Generator name: {generator_name}")
            print(f"Batch size: {batch_size}")
            print(f"Time taken: {time.time() - time_start:.3f} seconds")
            time_per_prompt = (time.time() - time_start) / batch_size
            print(f"Time per prompt: {time_per_prompt:.3f} seconds")
            time_per_prompt_dict[generator_name][batch_size] = round(time_per_prompt, 3)
            prompt_generator.release(empty_cuda_cache=True)

    max_columns = max(len(batch_sizes) for batch_sizes in time_per_prompt_dict.values())

    # Find the maximum length of model name for formatting
    max_model_length = max(len(model) for model in time_per_prompt_dict.keys())

    # Print the headers
    print(f'{"Model":<{max_model_length}}\t', end="")
    for i in range(0, max_columns):
        print(f"{2**i}\t\t", end="")
    print()

    # Print each row of the table
    for model, batch_size_times in time_per_prompt_dict.items():
        print(f"{model:<{max_model_length}}\t", end="")
        for _, time_per_prompt in batch_size_times.items():
            print(f"{time_per_prompt}\t\t", end="")
        print()
