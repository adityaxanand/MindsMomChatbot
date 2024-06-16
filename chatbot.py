from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model and tokenizer
tokenizer = BertTokenizer.from_pretrained('saved_model')
model = BertForSequenceClassification.from_pretrained('saved_model')

# Initialize Flask app
app = Flask(__name__)

# Define the sentiment analysis function
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return 'positive' if prediction == 1 else 'negative'

# Define the chatbot response function
def get_response(sentiment):
    if sentiment == 'positive':
        return "I'm glad to hear that you're feeling good! Keep it up!"
    elif sentiment == 'negative':
        return "I'm sorry that you're feeling down. It's important to talk about it. How can I help?"
    else:
        return "I see. Tell me more about it."

# Define the chatbot endpoint
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    sentiment = analyze_sentiment(user_input)
    response = get_response(sentiment)
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000)
