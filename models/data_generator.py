import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataGenerator(Sequence):

    def __init__(self, descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):

        self.descriptions = descriptions
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.image_ids = list(descriptions.keys())


    def __len__(self):
        return len(self.image_ids) // self.batch_size
    
    def __getitem__(self, index):

        X1, X2, y = [], [], []

        batch_ids = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]

        for image_id in batch_ids:

            if image_id not in self.features:
                continue

            feature = self.features[image_id][0]

            for caption in self.descriptions[image_id]:

                seq = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):

                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences(
                        [in_seq],
                        maxlen=self.max_length,
                        padding='post'
                    )[0]

                    out_seq = to_categorical(
                        [out_seq],
                        num_classes=self.vocab_size
                    )[0]

                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)

        return (
    {
        "image": np.array(X1),
        "text": np.array(X2)
    },
    np.array(y)
)
    


