import matplotlib.pyplot as plt
import numpy as np

from src.utils import *
from src.patchmatch import *
import cv2


if __name__ == "__main__":
    img = cv2.imread("/content/input.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = white2black(img)

    p_size = 7
    itr = 2

    hl, lr = bounding_box(img, 10, 0)
    src = img[hl[1]: lr[1], hl[0]:lr[0], :]

    f, dist, hist = patchmatch(src, img, hl, lr, 1, alpha=0.5)

    test = [reconstruction(ann, img).astype("int") for ann in hist]

    multi_plot([3, 2], test, ["Initial", "1/4 Iter", "1/2 Iter", "3/4 Iter", "Iter 1", "Iter 2"], "PatchMatch", (20, 15))
