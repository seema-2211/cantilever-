import pickle

from models.predict_caption import load_cnn, extract_feature, generate_caption
from models.caption_model import build_model
from models.load_captions import load_descriptions
from models.tokenizer_builder import create_tokenizer
from models.max_length import max_length
from tensorflow.keras.models import load_model

captions = load_descriptions("dataset/Flickr8k/Flickr8k.token.txt")

tokenizer = create_tokenizer(captions)

vocab_size = len(tokenizer.word_index) + 1

max_len = max_length(captions)

features = pickle.load(open("dataset/features/features.pkl", "rb"))


model = build_model(vocab_size, max_len)
model.load_weights(
    "models/image_captioning_model.keras"
)


cnn = load_cnn()


image_path = "dataset/Flickr8k/Images/2391812384_7429b5e567.jpg"  

image_feature = extract_feature(image_path, cnn)

caption = generate_caption(model, tokenizer, image_feature, max_len)

print("\nRAW CAPTION:")
print(repr(caption)) 
print("\nGenerated Caption:")
print(caption)
