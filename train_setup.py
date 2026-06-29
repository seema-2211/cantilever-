import pickle

from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer
from models.max_length import max_length

# captions
captions = load_descriptions("dataset/Flickr8k/Flickr8k.token.txt")

# tokenizer
tokenizer = create_tokenizer(captions)

# vocab size
vocab_size = len(tokenizer.word_index) + 1

# max length
max_len = max_length(captions)

# load features
features = pickle.load(open("dataset/features/features.pkl", "rb"))

print("VOCAB SIZE:", vocab_size)
print("MAX LENGTH:", max_len)
print("FEATURES LOADED:", len(features))
