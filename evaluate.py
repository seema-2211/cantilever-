import pickle

from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.models import load_model
from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer
from models.max_length import max_length
from models.caption_model import build_model
from models.predict_caption import generate_caption

# Load captions
captions = load_descriptions(
    "dataset/Flickr8k/Flickr8k.token.txt"
)

tokenizer = create_tokenizer(captions)

vocab_size = len(tokenizer.word_index) + 1
max_len = max_length(captions)

# Load features
features = pickle.load(
    open("dataset/features/features.pkl", "rb")
)



model = load_model(
    "models/image_captioning_model.keras",
    compile=False
)
model.save(
    "models/image_captioning_model.keras"
)
actual = []
predicted = []

count = 0

for image_id, desc_list in captions.items():

    if image_id not in features:
        continue

    yhat = generate_caption(
        model,
        tokenizer,
        features[image_id],
        max_len
    )

    references = [
        d.split()
        for d in desc_list
    ]

    actual.append(references)
    predicted.append(yhat.split())

    count += 1

    if count >= 100:
        break

print("Evaluated:", count, "images")

print(
    "BLEU-1:",
    corpus_bleu(
        actual,
        predicted,
        weights=(1.0, 0, 0, 0)
    )
)

print(
    "BLEU-2:",
    corpus_bleu(
        actual,
        predicted,
        weights=(0.5, 0.5, 0, 0)
    )
)

print(
    "BLEU-3:",
    corpus_bleu(
        actual,
        predicted,
        weights=(0.33, 0.33, 0.33, 0)
    )
)

print(
    "BLEU-4:",
    corpus_bleu(
        actual,
        predicted,
        weights=(0.25, 0.25, 0.25, 0.25)
    )
)