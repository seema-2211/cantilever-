from tensorflow.keras.preprocessing.text import Tokenizer

def create_tokenizer(descriptions):

    lines = []

    for captions in descriptions.values():
        lines.extend(captions)

    tokenizer = Tokenizer(
        oov_token="<unk>"   # handles unknown words
    )

    tokenizer.fit_on_texts(lines)

    return tokenizer