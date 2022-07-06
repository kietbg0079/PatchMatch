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



def inBox(height, width, box_hl, box_lr):
    if box_hl[0] <= width <= box_lr[0] and box_hl[1] <= height <= box_lr[1]:
        return True
    return False


def gen_mask(img, mask_value=255):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.ones_like(gray_img)
    mask[gray_img == mask_value] = 0
    return mask


def isHole(mask, y, x, padding=7):
    # Hole is non-black pixel
    if mask[y, x] == 0:
        return False
    return True


def white2black(img):
    h, w, d = img.shape
    for i in range(h):
        for j in range(w):
            low = np.array([240, 240, 240]) <= img[i, j, :]
            high = img[i, j, :] <= np.array([255, 255, 255])
            if (low == high).all() and (low == np.array([True, True, True])).all():
                img[i, j, :] = np.array([0, 0, 0])

    return img


def bounding_box(img, pad=3, col=0):
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
            if gray[i][j] == col:

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


def reconstruction(f, ref):
    fh, fw = f.shape

    temp = np.zeros((fh, fw, 3))

    for y in range(fh):
        for x in range(fw):
            temp[y, x, :] = ref[f[y, x][0], f[y, x][1], :]
    return temp


def cal_distance(a, b, src_pad, ref_pad, p_size):
    p = p_size // 2

    patch_a = src_pad[a[0]:a[0] + p_size, a[1]:a[1] + p_size, :]
    patch_b = ref_pad[b[0]:b[0] + p_size, b[1]:b[1] + p_size, :]

    # holeCount = 0
    # ans = 0

    # for j in range(0, p_size):
    #   for i in range(0, p_size):
    #     if all(patch_a[j, i] == np.array([0, 0, 0])):
    #       holeCount += holeCount
    #       continue
    #     ans += np.sum(np.square(patch_a[j, i] - patch_b[j, i]))

    # dist = ans / (p_size**2 - holeCount)

    dist = np.sum(np.square(patch_a - patch_b))
    return dist



