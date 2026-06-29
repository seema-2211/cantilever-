import pickle
from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer
from models.max_length import max_length
from models.data_generator import data_generator

captions = load_descriptions("dataset/Flickr8k/Flickr8k.token.txt")

tokenizer = create_tokenizer(captions)

vocab_size = len(tokenizer.word_index) + 1

max_len = max_length(captions)

features = pickle.load(open("dataset/features/features.pkl", "rb"))
  

gen = data_generator(
    captions,
    features,
    tokenizer,
    max_len,
    vocab_size
)

print("Generator created successfully!")
