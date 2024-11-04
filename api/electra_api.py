from flask import Blueprint, request, jsonify
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch

electra_api = Blueprint('electra_api', __name__)

checkpoint = "models/AImodel"  
model = ElectraForSequenceClassification.from_pretrained(checkpoint)
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

@electra_api.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return jsonify({'predicted_class': predicted_class})
