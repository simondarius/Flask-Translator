from flask import Flask, render_template, request
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import re

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("simondarius/eng_to_fr_translation")
model = TFAutoModelForSeq2SeqLM.from_pretrained("simondarius/eng_to_fr_translation")

def translate(input_text):
    inputs = tokenizer(input_text, return_tensors="tf")
    output_sequences = model.generate(inputs['input_ids'])
    translated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    cleaned_text = re.sub(r'& #160;|&nbsp;', ' ', translated_text).strip()
    return cleaned_text

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""
    if request.method == 'POST':
        input_text = request.form['input_text']
        translated_text = translate(input_text)
    return render_template('translate.html', translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)