import string

def clean_caption(text):

    text = text.lower()

    text = ''.join(
        ch for ch in text
        if ch not in string.punctuation
    )

    words = text.split()

    words = [
        word
        for word in words
        if len(word) > 1 and word.isalpha()
    ]

    return ' '.join(words)