import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


def load_cnn():

    model = InceptionV3(weights="imagenet")
    model = Model(inputs=model.input, outputs=model.layers[-2].output)

    return model


def extract_feature(image_path, cnn_model):

    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature = cnn_model.predict(img, verbose=0)

    return feature


def generate_caption(model, tokenizer, image_feature, max_length):

    in_text = "startseq"

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="post",
            truncating="post"
        )

        yhat = model.predict([image_feature, sequence], verbose=0)

        yhat = np.argmax(yhat[0])

        word = tokenizer.index_word.get(yhat)
        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text