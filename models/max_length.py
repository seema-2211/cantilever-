def max_length(descriptions):

    return max(len(caption.split()) for captions in descriptions.values() for caption in captions)
