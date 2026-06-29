from flask import Flask, render_template, request
import os
import pickle

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

from models.predict_caption import extract_feature, generate_caption

from models.caption_model import build_model
from tensorflow.keras.models import load_model



app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

captions = load_descriptions(
"dataset/Flickr8k/Flickr8k.token.txt"
)
tokenizer = pickle.load(
    open("models/tokenizer.pkl", "rb")
)
vocab_size = len(tokenizer.word_index) + 1

max_len = 23

model = build_model(
vocab_size,
max_len
)

model = load_model(
    "models/image_captioning_model.keras"
)


cnn_model = InceptionV3(
weights="imagenet"
)

cnn_model = Model(
inputs=cnn_model.inputs,
outputs=cnn_model.layers[-2].output
)

@app.route("/", methods=["GET", "POST"])
def home():
    caption = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(filepath)

            feature = extract_feature(
                filepath,
                cnn_model
            )

            caption = generate_caption(
                model,
                tokenizer,
                feature,
                max_len
            )

            image_path = filepath

    return render_template(
        "index.html",
        caption=caption,
        image_path=image_path
    )



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port
    )
