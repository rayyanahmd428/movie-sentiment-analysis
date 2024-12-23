from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import logging
import re
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Set logging level
logging.basicConfig(level=logging.INFO)

# Load the model
logging.info('Loading model...')
model = tf.keras.models.load_model('lstm_model_acc_0.834.keras')
logging.info('Model loaded successfully.')

# Load the tokenizer
logging.info('Loading tokenizer...')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
logging.info('Tokenizer loaded successfully.')

# Compile the pattern for removing HTML tags
TAG_RE = re.compile(r'<[^>]+>')

# Function to remove HTML tags
def remove_tags(text):
    return TAG_RE.sub('', text)

# Function to clean and preprocess the text
def preprocess_text(sen):
    """Clean text data, leaving only 2 or more character long non-stopwords
    composed of A-Z & a-z only in lowercase."""

    sentence = sen.lower()

    # Remove HTML tags
    sentence = remove_tags(sentence)

    # Remove punctuation and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Remove single characters
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    sentence = ' '.join(word for word in sentence.split() if word not in stop_words)

    logging.info(f'Cleaned text: {sentence}')
    return sentence

# Function to tokenize and pad text
def tokenize_and_pad(text):
    max_length = 100  # Ensure this matches the model's expected input length
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    logging.info(f'Tokenized text: {sequences}')
    logging.info(f'Padded sequences: {padded_sequences}')
    return np.array(padded_sequences)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    cleaned_text = preprocess_text(text)
    tokenized_and_padded_text = tokenize_and_pad(cleaned_text)
    prediction = model.predict(tokenized_and_padded_text)[0][0]
    logging.info(f'Model output: {prediction}')
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)



