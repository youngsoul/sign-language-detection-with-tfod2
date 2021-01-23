# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import logging
import glob
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

def augment_image(image_path, count, output_path, prefix=""):
    # load the input image, convert itto a numpy array and then reshape it to have an extra dimension
    logging.debug(f"loading example image: {image_path}")
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # construct the image generator for the data augmentation then
    # initialize the total number of images generated thus far
    # TODO Edit these values to adjust how images are generated
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    total = 0

    # construct the actual python generator
    logging.debug("generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=output_path, save_prefix=prefix, save_format="jpg")

    for image in imageGen:
        # increment counter
        total += 1

        # if we have reach the specifiedc number of examples, break from the loop
        if total == count:
            break


if __name__ == '__main__':
    # construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images-dir", required=True, help="path to the input images")
    ap.add_argument("-o", "--output-dir", required=True, help="path to the output directory to store augmentations examples")
    ap.add_argument("--per-image", type=int, default=4, help="# of training samples to generate per image")
    ap.add_argument("--prefix", type=str, required=False, default="", help="prefix to add to all generated images")
    ap.add_argument("--image-type", type=str, required=False, default="jpg", help="Image type. E.g. [jpg, png, tiff, etc]")

    args = vars(ap.parse_args())

    Path(args['output_dir']).mkdir(parents=True, exist_ok=True)

    for base_image in glob.glob(args['images_dir'] + f"/*.{args['image_type']}"):
        augment_image(base_image, args['per_image'], args['output_dir'], prefix=args['prefix'])

