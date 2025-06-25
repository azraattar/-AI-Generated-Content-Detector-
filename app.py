from flask import Flask, render_template, request
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load AI detection model and tokenizer once
model_name = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Advanced AI detection using pre-trained model
def detect_ai_content(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    labels = ['Human', 'AI-generated']
    result = dict(zip(labels, probs.detach().numpy()[0]))
    
    prediction = max(result, key=result.get)
    confidence = round(result[prediction] * 100, 2)
    
    return f"{'✅' if prediction == 'Human' else '⚠️'} {prediction} (Confidence: {confidence}%)"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Run AI detection
        result = detect_ai_content(text)

        return render_template('result.html', result=result, filename=file.filename)

    return "Something went wrong", 500

if __name__ == '__main__':
    app.run(debug=True)
