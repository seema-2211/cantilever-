from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer

captions = load_descriptions("dataset/Flickr8k/Flickr8k.token.txt")

tokenizer = create_tokenizer(captions)

print("Total Words:", len(tokenizer.word_index))
print("Most common index:", tokenizer.word_index.get("startseq"))