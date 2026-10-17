from exllamav3 import Config, Model, Cache, Tokenizer, Generator

#model_dir = "/home/andrey/models/test"
model_dir = "/mnt/models/qwen3.8-27b"
#model_dir = "/home/andrey/models/test2"
#model_dir = "/mnt/models/qwen3.6-35b"
prompt    = "What is the capital of France?"

def main():
    config = Config.from_directory(model_dir)
    model  = Model.from_config(config)
    cache  = Cache(model, max_num_tokens=1024)
    model.load(tensor_p=True, tp_backend="native", verbose=True, progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    response = generator.generate(prompt=prompt, max_new_tokens=256)
    
    print(response)

    model.unload()

if __name__ == '__main__':
    main()