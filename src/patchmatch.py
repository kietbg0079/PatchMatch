import numpy as np

from .utils import *



def cal_distance(a, b, src_pad, ref_pad, p_size):
    patch_a = src_pad[a[0]:a[0] + p_size, a[1]:a[1] + p_size, :]
    patch_b = ref_pad[b[0]:b[0] + p_size, b[1]:b[1] + p_size, :]

    temp = patch_b - patch_a
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num

    return dist

def initialization(src, ref, hl, lr, p_size, patch=3):
    srch, srcw, srcd = src.shape
    refh, refw, refd = ref.shape

    p = p_size // 2

    ref_padding = np.ones([refh + p * 2, refw + p * 2, 3]) * np.nan
    ref_padding[p: refh + p, p: refw + p, :] = ref

    src_padding = np.ones([srch + p * 2, srcw + p * 2, 3]) * np.nan
    src_padding[p: srch + p, p: srcw + p, :] = src

    f = np.zeros([srch, srcw], dtype=object)
    dist = np.zeros([srch, srcw])

    for y in range(srch):
        for x in range(srcw):
            a = np.array([y, x])

            by = np.random.randint(0, refh - 1)
            bx = np.random.randint(0, refw - 1)
            while inBox(by, bx, hl, lr):
                by = np.random.randint(0, refh - 1)
                bx = np.random.randint(0, refw - 1)

            f[y, x] = np.array([by, bx])
            dist[y, x] = cal_distance(a, f[y, x], src_padding, ref_padding, p_size)

    return f, dist, src_padding, ref_padding


def propagation(a, f, dist, src_pad, ref_pad, p_size, is_odd, hl, lr):
    srch = np.size(src_pad, 0) - p_size + 1
    srcw = np.size(src_pad, 1) - p_size + 1

    y = a[0]
    x = a[1]

    if is_odd:

        d_left = dist[max(y - 1, 0), x]
        d_up = dist[y, max(x - 1, 0)]
        d_current = dist[y, x]
        idx = np.argmin(np.array([d_current, d_left, d_up]))

        if idx == 1:
            py = f[max(y - 1, 0), x][0]
            px = f[max(y - 1, 0), x][1]

            if not inBox(py, px, hl, lr):
                f[y, x] = f[max(y - 1, 0), x]
                dist[y, x] = cal_distance(a, f[y, x], src_pad, ref_pad, p_size)

        if idx == 2:
            py = f[y, max(x - 1, 0)][0]
            px = f[y, max(x - 1, 0)][1]

            if not inBox(py, px, hl, lr):
                f[y, x] = f[y, max(x - 1, 0)]
                dist[y, x] = cal_distance(a, f[y, x], src_pad, ref_pad, p_size)

    else:

        d_right = dist[min(y + 1, srch - 1), x]
        d_down = dist[y, min(x + 1, srcw - 1)]
        d_current = dist[y, x]
        idx = np.argmin(np.array([d_current, d_right, d_down]))

        if idx == 1:
            py = f[min(y + 1, srch - 1), x][0]
            px = f[min(y + 1, srch - 1), x][1]

            if not inBox(py, px, hl, lr):
                f[y, x] = f[min(y + 1, srch - 1), x]
                dist[y, x] = cal_distance(a, f[y, x], src_pad, ref_pad, p_size)

        if idx == 2:
            py = f[y, min(x + 1, srcw - 1)][0]
            px = f[y, min(x + 1, srcw - 1)][1]

            if not inBox(py, px, hl, lr):
                f[y, x] = f[y, min(x + 1, srcw - 1)]
                dist[y, x] = cal_distance(a, f[y, x], src_pad, ref_pad, p_size)


def random_search(a, f, dist, src_pad, ref_pad, p_size, hl, lr, alpha=0.5):
    y = a[0]
    x = a[1]

    refh, refw, refd = ref_pad.shape

    i = 0

    search_h = refh * alpha ** i
    search_w = refw * alpha ** i
    b_y = f[y, x][0]
    b_x = f[y, x][1]
    while search_h > 1 and search_w > 1:

        search_min_r = max(b_y - search_h, 0)
        search_max_r = min(b_y + search_h, refh - p_size)

        search_min_c = max(b_x - search_w, 0)
        search_max_c = min(b_x + search_w, refw - p_size)

        random_b_y = np.random.randint(search_min_r, search_max_r)
        random_b_x = np.random.randint(search_min_c, search_max_c)
        while inBox(random_b_y, random_b_x, hl, lr):
            random_b_y = np.random.randint(search_min_r, search_max_r)
            random_b_x = np.random.randint(search_min_c, search_max_c)

        search_h = refh * alpha ** i
        search_w = refw * alpha ** i

        b = np.array([random_b_y, random_b_x])
        d = cal_distance(a, b, src_pad, ref_pad, p_size)
        if d < dist[y, x]:
            dist[y, x] = d
            f[y, x] = b

        i += 1


def NNS(src, ref, p_size, hl, lr, itr):
    srch, srcw, srcd = src.shape

    f, dist, src_pad, ref_pad = initialization(src, ref, hl, lr, p_size)

    for itr in range(1, itr + 1):
        if itr % 2 == 0:
            for y in range(srch - 1, -1, -1):
                for x in range(srcw - 1, -1, -1):
                    a = np.array([y, x])
                    propagation(a, f, dist, src_pad, ref_pad, p_size, False, hl, lr)
                    random_search(a, f, dist, src_pad, ref_pad, p_size, hl, lr)
        else:
            for y in range(srch):
                for x in range(srcw):
                    a = np.array([y, x])
                    propagation(a, f, dist, src_pad, ref_pad, p_size, True, hl, lr)
                    random_search(a, f, dist, src_pad, ref_pad, p_size, hl, lr)

        print("iteration: %d" % (itr))
    return f