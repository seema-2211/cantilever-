from models.caption_model import build_model
from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer
from models.max_length import max_length

captions = load_descriptions("dataset/Flickr8k/Flickr8k.token.txt")

tokenizer = create_tokenizer(captions)

vocab_size = len(tokenizer.word_index) + 1

max_len = max_length(captions)

model = build_model(vocab_size, max_len)

model.summary()
