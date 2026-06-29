from models.load_captions import load_descriptions
from models.build_vocab import build_vocabulary

captions = load_descriptions(
    "dataset/Flickr8k/Flickr8k.token.txt"
)

vocab = build_vocabulary(captions)

print("Vocabulary Size:", len(vocab))