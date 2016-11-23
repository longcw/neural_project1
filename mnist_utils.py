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


def save_npz(save_list, sess, name='model.npz'):
    """Input parameters and the file name, save parameters into .npz file. Use tl.utils.load_npz() to restore.
    """

    # save params into a list
    save_list_var = []
    for k, value in enumerate(save_list):
        save_list_var.append(sess.run(value))

    np.savez(name, params=save_list_var)
    print('Model is saved to: %s' % name)

    return save_list_var

    ## save params into a dictionary
    # rename_dict = {}
    # for k, value in enumerate(save_dict):
    #     rename_dict.update({'param'+str(k) : value.eval()})
    # np.savez(name, **rename_dict)
    # print('Model is saved to: %s' % name)


def load_npz(name):
    d = np.load(name)

    return d['params']
