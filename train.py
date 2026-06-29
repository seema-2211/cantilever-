import pickle
import numpy as np

from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer
from models.max_length import max_length
from models.caption_model import build_model
from models.data_generator import DataGenerator

captions = load_descriptions("dataset/Flickr8k/Flickr8k.token.txt")

tokenizer = create_tokenizer(captions)

vocab_size = len(tokenizer.word_index) + 1

max_len = max_length(captions)

features = pickle.load(open("dataset/features/features.pkl", "rb"))

print("Dataset loaded")
print("Vocab size:", vocab_size)
print("Max length:", max_len)

model = build_model(vocab_size, max_len)

print(model.summary())

from models.data_generator import DataGenerator

generator = DataGenerator(
    captions,
    features,
    tokenizer,
    max_len,
    vocab_size,
    batch_size=32
)


steps = len(captions)

model.fit(
    generator,
    epochs=30,
    verbose=1
)
model.save("models/image_captioning_model.keras")

print("Model saved successfully!")
