from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def create_sequences(tokenizer, max_length, descriptions, vocab_size):

    X1, X2, y = [], [], []

    for key, captions_list in descriptions.items():

        for caption in captions_list:

            seq = tokenizer.texts_to_sequences([caption])[0]

            for i in range(1, len(seq)):

                in_seq = seq[:i]
                out_seq = seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                out_seq = np.eye(vocab_size)[out_seq]

                X1.append(key)
                X2.append(in_seq)
                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)
