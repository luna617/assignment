from fastapi import FastAPI
from llama_cpp import Llama
from transformers import pipeline
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from asyncer import asyncify

# import uvicorn


class GenerationRequest(BaseModel):
    text: str
    temperature: float | None = 0.3
    max_tokens: float | None = 1000
    top_p: float | None = 0.1


instruct_model = Llama(model_path=r"<MODEL-PATH>")

translation_model = pipeline("translation", model=r"<MODEL-PATH>", max_length=1000)
app = FastAPI()


@app.post("/summarize")
async def summarize(generation_request: GenerationRequest):
    translated_text = await asyncify(translate)(generation_request.text, src_lang='heb_Hebr', tgt_lang='eng_Latn')
    print("Finished translating")

    generation_request = {
        "text": f"Consider the following text: {translated_text}.\nA 5 points summary for the text is: ",
        "temperature": generation_request.temperature,
        "max_tokens": generation_request.max_tokens,
        "top_p": generation_request.top_p
    }

    generation_request = GenerationRequest(**generation_request)
    return StreamingResponse(five_points_summary(generation_request), media_type='text/event-stream')


def translate(text: str, src_lang: str, tgt_lang: str):
    res = translation_model(text, src_lang=src_lang, tgt_lang=tgt_lang)[0]["translation_text"]
    return res


async def five_points_summary(generation_request: GenerationRequest):
    print(generation_request)
    res = instruct_model.create_chat_completion(
        temperature=generation_request.temperature,
        max_tokens=generation_request.max_tokens,
        top_p=generation_request.top_p,
        messages=[
            {
                "role": "user",
                "content": generation_request.text
            }
        ],
        stream=True)

    current_line = ""
    for chunk in res:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            current_line += delta['content']
            print(delta['content'])
            if delta['content'].endswith('\n'):
                translated_current_line = await asyncify(translate)(current_line, src_lang='eng_Latn', tgt_lang='heb_Hebr')
                print(translated_current_line)
                yield translated_current_line
                current_line = ""

    translated_current_line = await asyncify(translate)(current_line, src_lang='eng_Latn', tgt_lang='heb_Hebr')
    print(translated_current_line)
    yield translated_current_line


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)