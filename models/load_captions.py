from models.preprocess_captions import clean_caption

def load_descriptions(filename):

    descriptions = {}

    with open(filename, "r") as file:

        for line in file:

            line = line.strip()

            if len(line) == 0:
                continue

            parts = line.split('\t')

            if len(parts) != 2:
                continue

            image_info, caption = parts

            image_id = image_info.split('#')[0]
            image_id = image_id.split('.')[0]

            caption = clean_caption(caption)

            caption = f"startseq {caption} endseq"

            descriptions.setdefault(
                image_id,
                []
            ).append(caption)

    return descriptions