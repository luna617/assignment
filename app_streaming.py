from fastapi import FastAPI
from llama_cpp import Llama
from transformers import pipeline
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import Optional

# import uvicorn


class Message(BaseModel):
    text: str
    temperature: Optional[float] = 0.3
    max_tokens: Optional[float] = 100
    top_p: Optional[float] = 0.1

# echo = False
# stop = ["Q", "\n"]


instruct_model = Llama(model_path=r"<MODEL_PATH>")

translation_model = pipeline("translation", model=r"<MODEL_PATH>", max_length=1000)
app = FastAPI()


@app.post("/summarize")
async def summarize(message: Message):
    translated_text = await translate(message.text, src_lang='heb_Hebr', tgt_lang='eng_Latn')
    print("Finished translating")
    message.text = f"Consider the following text: {translated_text}.\nA 5 points summary for the text is: "
    return StreamingResponse(five_points_summary(message), media_type='text/event-stream')


async def translate(text: str, src_lang: str, tgt_lang: str):
    res = translation_model(text, src_lang=src_lang, tgt_lang=tgt_lang)[0]["translation_text"]
    return res


async def five_points_summary(message: Message):
    print(message)
    res = instruct_model.create_chat_completion(
        temperature=message.temperature,
        max_tokens=message.max_tokens,
        top_p=message.top_p,
        messages=[
            {
                "role": "user",
                "content": message.text
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
                # print(translated_current_line)
                yield translated_current_line
                current_line = ""

    translated_current_line = await translate(current_line, src_lang='eng_Latn', tgt_lang='heb_Hebr')
    # print(translated_current_line)
    yield translated_current_line


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)