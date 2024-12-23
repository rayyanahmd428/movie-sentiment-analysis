import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
nltk.download('stopwords')

# Load your dataset
texts = ["Load your dataset of texts here"]

# Parameters
vocab_size = 10000  # Adjust as needed

# Create a tokenizer and fit it on the texts
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)

# Save the tokenizer to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer created and saved as 'tokenizer.pickle'.")
