import os
from PIL import Image

from resizeimage import resizeimage

files = os.listdir("./Img_Samsung")

for img in files:
    with open(f"./Img_Samsung/{img}", 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [200, 100])
            cover.save(f"./Img_Sam/{img}", image.format)