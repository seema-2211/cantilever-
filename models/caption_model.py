from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model

def build_model(vocab_size, max_length):

    # IMAGE INPUT
    image_input = Input(name="image", shape=(2048,))
    fe1 = Dropout(0.4)(image_input)
    fe2 = Dense(256, activation='relu')(fe1)

    text_input = Input(name="text", shape=(max_length,))

    se1 = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)

    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[image_input, text_input], outputs=outputs)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam'
    )

    return model
