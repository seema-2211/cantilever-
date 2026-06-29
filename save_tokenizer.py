import pickle

from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer

captions = load_descriptions(
    "dataset/Flickr8k/Flickr8k.token.txt"
)

tokenizer = create_tokenizer(captions)

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved.")

