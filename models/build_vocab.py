def build_vocabulary(descriptions):

    vocabulary = set()

    for captions in descriptions.values():

        for caption in captions:

            vocabulary.update(
                caption.split()
            )

    return vocabulary
