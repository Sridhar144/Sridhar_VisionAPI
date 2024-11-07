# import os, io
# from google.cloud import vision
# from google.cloud.vision import types

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'service_account_token.json'

# client = vision.ImageAnnotatorClient()

import io, os, argparse
from numpy import random
from google.cloud import vision_v1
from Pillow_Utility import draw_borders, Image
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"peppy-citron-421415-d6e0bc494625.json"
client = vision_v1.ImageAnnotatorClient()

file_name = 'download.jpeg'
image_path = os.path.join('./Images', file_name)

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = vision_v1.types.Image(content=content)
response = client.object_localization(image=image)
localized_object_annotations = response.localized_object_annotations

pillow_image = Image.open(image_path)
df = pd.DataFrame(columns=['name', 'score'])
for obj in localized_object_annotations:
    df = df._append(
        dict(
            name=obj.name,
            score=obj.score
        ),
        ignore_index=True)
    
    r, g, b = random.randint(150, 255), random.randint(
        150, 255), random.randint(150, 255)

    draw_borders(pillow_image, obj.bounding_poly, (r, g, b),
                 pillow_image.size, obj.name, obj.score)

print(df)
pillow_image.show()

def main(image_file):
    # Instantiates a client

    # Loads the image into memory
    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('Labels:')
    for label in labels:
        print(label.description)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('Images/download.jpeg', help='The image you\'d like to label.')
    args = parser.parse_args()
    main(args.image_file)