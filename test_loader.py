from models.load_captions import load_descriptions

captions = load_descriptions(
    "dataset/Flickr8k/Flickr8k.token.txt"
)

print("Images:", len(captions))

first_key = list(captions.keys())[0]

print(first_key)

print(captions[first_key][0])