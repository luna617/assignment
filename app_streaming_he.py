from fastapi import FastAPI
from llama_cpp import Llama
from transformers import pipeline
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import uvicorn


class Message(BaseModel):
    text: str


max_tokens = 100
temperature = 0.3
top_p = 0.1
echo = False
stop = ["Q", "\n"]

instruct_model = Llama(model_path=r"C:\Users\GH6738\models\Phi-3-mini-4k-instruct\Phi-3-mini-4k-instruct-F16.gguf")

translation_model = pipeline("translation", model=r"C:\Users\GH6738\models\nllb-200-distilled-600M",
                             src_lang='heb_Hebr', tgt_lang='eng_Latn', max_length=1000)
app = FastAPI()


@app.post("/summarize")
async def summarize(message: Message):
    translated_text = await translate(message.text)
    # print(translated_text)
    print("finished translating")
    prompt = f"Consider the following text: {translated_text}.\nA 5 points summary for the text is: "
    # print(prompt)
    return StreamingResponse(five_points_summary(prompt), media_type='text/event-stream')


async def translate(text: str, src_lang='heb_Hebr', tgt_lang='eng_Latn'):
    res = translation_model(text, src_lang=src_lang, tgt_lang=tgt_lang)[0]["translation_text"]
    return res


async def five_points_summary(prompt: str):
    res = instruct_model.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True)

    current_line = ""
    for chunk in res:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            current_line += delta['content']
            if delta['content'] == '\n':
                translated_current_line = await translate(current_line, src_lang='eng_Latn', tgt_lang='heb_Hebr')
                print(translated_current_line)
                yield translated_current_line
                current_line = ""

    translated_current_line = await translate(current_line, src_lang='eng_Latn', tgt_lang='heb_Hebr')
    print(translated_current_line)
    yield translated_current_line

#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)