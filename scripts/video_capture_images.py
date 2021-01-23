import cv2
import glob
import uuid
import time
from pathlib import Path
import argparse
import model_config as mc
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

"""
Script will capture images from the computer webcam.

Press the '.' key to capture an image.  
      If --auto-capture option is used the script will automatically collect a new image every second
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="label to associate with collected image")
    ap.add_argument("--num-images", required=False, default=15, type=int, help="Number of images to collect with the given label")
    ap.add_argument("--auto-capture", action='store_true', help="Flag to automatically capture image" )

    args = vars(ap.parse_args())
    label = args["label"]

    IMAGES_PATH = mc.COLLECTED_IMAGES
    Path(IMAGES_PATH).mkdir(parents=True, exist_ok=True)

    number_imgs = args['num_images']

    cap = cv2.VideoCapture(0)
    time.sleep(2)

    ret, frame = cap.read()
    cv2.imshow("Image", frame)
    cv2.waitKey(1)

    image_dir = f"{IMAGES_PATH}"
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    print(f"Collecting images for {label}")
    if not args['auto_capture']:
        print("To capture an image, press the '.' key.")
    print("Getting ready...")
    print("5")
    time.sleep(1)
    print("4")
    time.sleep(1)
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    imgnum = 0
    while imgnum < number_imgs:
        ret, frame = cap.read()
        cv2.imshow("Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('.') or args['auto_capture']:
            msg = f"{label} Picture {imgnum+1} of {number_imgs}"
            print(msg)
            Path(f"{image_dir}/{label}").mkdir(parents=True, exist_ok=True)
            imagename = f"{image_dir}/{label}/{label}.{str(uuid.uuid1())}.jpg"
            cv2.imwrite(imagename, frame)
            cv2.putText(frame, msg, (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Image", frame)
            cv2.waitKey(1)
            imgnum +=1
            print("Sleep")
            time.sleep(1)

    cap.release()
