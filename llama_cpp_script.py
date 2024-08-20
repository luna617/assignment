from llama_cpp import Llama


# Instanciate the model
my_aweseome_llama_model = Llama(model_path=r"C:\Users\GH6738\models\Phi-3-mini-4k-instruct\Phi-3-mini-4k-instruct-F16.gguf")


prompt = (f"Consider the following text: today is a very good day, the sun is shining and the sky is blue. "
          f"A 5 points summary for this text are: ")
max_tokens = 100
temperature = 0.3
top_p = 0.1
echo = False
stop = ["Q", "\n"]


# Define the parameters
# model_output = my_aweseome_llama_model(
#        prompt,
#        max_tokens=max_tokens,
#        temperature=temperature,
#        top_p=top_p,
#        echo=echo,
#        stop=stop,
#    )
# final_result = model_output["choices"][0]["text"].strip()

# print(final_result)


output = my_aweseome_llama_model.create_chat_completion(
    messages=[
        { "role": "system", "content": "You are a story writing assistant." },
        {
            "role": "user",
            "content": "Write a story about llamas."
        }
    ],
    stream=True
)


for chunk in output:
    delta = chunk['choices'][0]['delta']
    if 'role' in delta:
        print(delta['role'], end=': ')
    elif 'content' in delta:
        print(delta['content'], end='')