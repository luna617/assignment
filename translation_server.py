# Use a pipeline as a high-level helper
import time

from transformers import pipeline


translation_model = pipeline("translation", model=r"C:\Users\GH6738\models\nllb-200-distilled-600M",
                             src_lang='heb_Hebr', tgt_lang="eng_Latn")
# pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M",
#                 src_lang='en', tgt_lang="he")



# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")


text = ["היום הוא יום יפה, השמש זורחת, הציפורים מצייצות והשמיים כחולים.",
"היום הוא יום יפה, השמש זורחת, הציפורים מצייצות והשמיים כחולים.",
"היום הוא יום יפה, השמש זורחת, הציפורים מצייצות והשמיים כחולים."]
# print(translation_model("היום הוא יום יפה, השמש זורחת, הציפורים מצייצות והשמיים כחולים."))

for out in translation_model(text, batch_size=1, truncation="only_first"):
    print(out)
    time.sleep(10)