from flask import Flask, request, render_template
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model_name = 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name,pad_token_id=tokenizer.eos_token_id)

model.eval()

def generate_text(prompt, max_length=300):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1,num_beams=5, no_repeat_ngram_size=2 ,early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    generated_text = generate_text(prompt)
    return render_template('index.html', prompt=prompt, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
