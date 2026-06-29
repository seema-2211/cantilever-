import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import pickle
def load_model():
    model = InceptionV3(weights="imagenet")
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return model

def extract_features(model, image_path):

    image = load_img(image_path, target_size=(299, 299))

    image = img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)


    return feature

def extract_all_features(directory):

    model = load_model()

    features = {}

    for img_name in tqdm(os.listdir(directory)):

        path = os.path.join(directory, img_name)

        try:
            feat = extract_features(model, path)
            features[img_name.split(".")[0]] = feat
        except:
            continue

    return features

def save_features(features, path="dataset/features/features.pkl"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(features, f)

if __name__ == "__main__":

    images_dir = "dataset/Flickr8k/Images"

    print("Extracting features...")

    features = extract_all_features(images_dir)

    print("Saving features...")

    save_features(features)

    print("Done! Features saved.")