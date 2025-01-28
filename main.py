from llama_cpp import Llama

llm = Llama(
      model_path="./SmolLM2.q8.gguf",
      n_gpu_layers=-1,
      seed=1337,
      n_ctx=2048,
      verbose=False
)

otp = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Name the planets in the solar system"
        },
    ],
    temperature=0.7,
)

print(otp["choices"][0]["message"])
