import matplotlib.pyplot as plt
import numpy as np

from src.utils import *
from src.patchmatch import *
import cv2


if __name__ == "__main__":
    img = cv2.imread("data/input.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = white2black(img)

    p_size = 3
    itr = 3

    hl, lr = bounding_box(img, 20)

    src = img[hl[1]: lr[1], hl[0]:lr[0], :]

    f = NNS(src, img, p_size, hl, lr, itr)

    result = reconstruction(f, img).astype("int")

    img[hl[1]: lr[1], hl[0]:lr[0], :] = result

    plt.imshow(img)
    plt.show()

