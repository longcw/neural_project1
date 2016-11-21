import numpy as np
import cv2
import os


def vector2image(x):
    x_image = np.reshape(x, [-1, 28, 28, 1])
    return x_image


def save_images(x, one_hot_labels, path):
    if len(x) <= 0:
        return False

    images = vector2image(x)
    labels = np.argmax(one_hot_labels, axis=1)
    for i, image in enumerate(images):
        filename = '{}_{}.jpg'.format(labels[i], i)
        filename = os.path.join(path, filename)
        cv2.imwrite(filename, image)
    return True
