from fastapi import FastAPI
from llama_cpp import Llama
from transformers import pipeline
from pydantic import BaseModel


class Message(BaseModel):
    text: str


max_tokens = 100
temperature = 0.3
top_p = 0.1
echo = False
stop = ["Q", "\n"]

instruct_model = Llama(model_path=r"C:\Users\GH6738\models\Phi-3-mini-4k-instruct\Phi-3-mini-4k-instruct-F16.gguf")

translation_model = pipeline("translation", model=r"C:\Users\GH6738\models\nllb-200-distilled-600M",
                             src_lang='heb_Hebr', tgt_lang="eng_Latn")
app = FastAPI()


@app.post("/summarize")
async def summarize(message: Message):
    translated_text = await translate(message.text)
    print(translated_text)
    prompt = f"Consider the following text: {translated_text}.\nA 5 points summary for the text is: "
    print(prompt)
    res = await five_points_summary(prompt)
    return res


async def translate(text):
    res = translation_model(text)[0]["translation_text"]
    return res


async def five_points_summary(prompt):
    res = instruct_model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
            stop=stop)["choices"][0]["text"].strip()

    return res


@app.get("/")
async def root():
    return {"message": "Hello World"}