import numpy as np
from PIL import Image

def load_image(file: str):
    img = Image.open(file)
    pix = img.load()

    width = img.size[0]
    height = img.size[1]

    _img = np.zeros((width, height), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            _img[i, j] = pix[j, i]

    return _img, width, height

def save_image(filtered, file):
    new_img = Image.fromarray(filtered.astype('uint8'), mode='L')
    new_img.save(file, format="BMP")