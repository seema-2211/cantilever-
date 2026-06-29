from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer
from models.max_length import max_length
from models.sequences import create_sequences

captions = load_descriptions("dataset/Flickr8k/Flickr8k.token.txt")

tokenizer = create_tokenizer(captions)

vocab_size = len(tokenizer.word_index) + 1

max_len = max_length(captions)

print("Vocab size:", vocab_size)
print("Max length:", max_len)

