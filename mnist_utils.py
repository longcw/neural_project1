import numpy as np
import cv2
import os
import shutil


def vector2image(x):
    x_image = np.reshape(x, [-1, 28, 28, 1])
    x_image[x_image != 0] = 255
    return x_image


def save_images(x, labels, predictions, path, one_hot=False):
    if len(x) <= 0:
        return False

    images = vector2image(x)
    if one_hot:
        labels = np.argmax(labels, axis=1)

    for i, image in enumerate(images):
        filename = '{}_{}_{}.jpg'.format(labels[i], predictions[i], i)
        filename = os.path.join(path, filename)
        cv2.imwrite(filename, image)
    return True


def mkdir(dir, overwrite=False):
    if os.path.isdir(dir):
        if overwrite:
            shutil.rmtree(dir)
        else:
            return
    os.mkdir(dir)