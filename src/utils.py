import numpy as np
import cv2
import matplotlib.pyplot as plt



def multi_plot(size, data, title, kind, figs=(10, 7)):
    fig = plt.figure(figsize=figs)
    plt.title(kind)
    for i in range(size[1]*size[0]):
        fig.add_subplot(size[1], size[0], i+1)

        plt.imshow(data[i], cmap ="gray")
        plt.axis('off')
        plt.title(title[i])

    plt.show()


def white2black(img):
    h, w, d = img.shape
    for i in range(h):
        for j in range(w):
            low = np.array([240, 240, 240]) <= img[i, j, :]
            high = img[i, j, :] <= np.array([255, 255, 255])
            if (low == high).all() and (low == np.array([True, True, True])).all():
                img[i, j, :] = np.array([0, 0, 0])

    return img



"""
    
    Finding bounding box of mask region
    
"""


def bounding_box(img, pad=3):
    # Check image channel
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Normalize gray image to 0-1 scale
    if np.max(gray) > 10:
        gray = gray / 255

    h, w = gray.shape

    idx = True

    for i in range(h):
        for j in range(w):
            if gray[i][j] == 0:

                if idx:
                    # Find the highest most left of bounding box
                    a = i
                    b = i
                    c = j
                    d = j
                    idx = False
                else:
                    if i > b:
                        # Find the lowest most left of bounding box
                        b = i
                    if j < c:
                        # Find the highest most right of bounding box
                        c = j
                    if j > d:
                        # Find the lowest most right of bounding box
                        d = j

    highest_most_left_point = (c - pad, a - pad)
    lowest_most_right_point = (d + pad, b + pad)

    return highest_most_left_point, lowest_most_right_point


def inBox(height, width, box_hl, box_lr):
    if box_hl[0] <= width <= box_lr[0] and box_hl[1] <= height <= box_lr[1]:
        return True
    return False


def reconstruction(f, ref):
    fh, fw = f.shape

    temp = np.zeros((fh, fw, 3))

    for y in range(fh):
        for x in range(fw):
            temp[y, x, :] = ref[f[y, x][0], f[y, x][1], :]
    return temp



