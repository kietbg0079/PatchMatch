import numpy as np

from .utils import *


def patchmatch(src, ref, hl, lr, iteration, padding=7, alpha=0.5):
    # Initial
    srch, srcw, srcd = src.shape
    refh, refw, refd = ref.shape

    p = padding // 2

    ref_padding = np.zeros([refh + p * 2, refw + p * 2, 3])
    ref_padding[p: refh + p, p: refw + p, :] = ref
    src_padding = np.zeros([srch + p * 2, srcw + p * 2, 3])
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
            dist[y, x] = cal_distance(a, f[y, x], src_padding, ref_padding, padding)

    print(f"Done initialization")

    for iter in range(iteration):

        print(f"Start {iter}")
        # Propagation
        ystart = 0
        yend = srch
        xstart = 0
        xend = srcw
        change = 1

        if iter % 2 == 0:
            ystart = srch - 1
            yend = -1
            xstart = srcw - 1
            xend = -1
            change = -1

            # Save image
        hist = []

        aok = int((yend - ystart) / 4)
        save_idx = [ystart, ystart + aok, ystart + 2 * aok, ystart + 3 * aok, yend - change]

        for col in range(ystart, yend, change):
            if col in save_idx:
                hist.append(np.copy(f))
            for row in range(xstart, xend, change):
                best_y, best_x = f[col, row]
                best_dist = dist[col, row]

                # Row propagate
                if row - change < srcw and row - change > 0:
                    yp = f[col, row - change][0]
                    xp = f[col, row - change][1] + change

                    if xp < refw and xp > 0 and not inBox(yp, xp, hl, lr):
                        distp = cal_distance(np.array([col, row]), np.array([yp, xp]), src_padding, ref_padding,
                                             padding)
                        if distp < best_dist:
                            best_y = yp
                            best_x = xp
                            best_dist = distp

                # Col propagate
                if col - change < srch and col - change > 0:
                    yp = f[col - change, row][0] + change
                    xp = f[col - change, row][1]

                    if yp < refh and yp > 0 and not inBox(yp, xp, hl, lr):
                        distp = cal_distance(np.array([col, row]), np.array([yp, xp]), src_padding, ref_padding,
                                             padding)

                        if distp < best_dist:
                            best_y = yp
                            best_x = xp
                            best_dist = distp

                # Random search
                i = 1
                rs_start = max(refh, refw)
                mag = rs_start * (alpha ** i)

                while mag > 1:

                    ymin = max(best_y - mag, 0)
                    ymax = min(best_y + mag + 1, refh)

                    xmin = max(best_x - mag, 0)
                    xmax = min(best_x + mag + 1, refw)

                    yp = np.random.randint(ymin, ymax)
                    xp = np.random.randint(xmin, xmax)

                    if not inBox(yp, xp, hl, lr):

                        distp = cal_distance(np.array([col, row]), np.array([yp, xp]), src_padding, ref_padding,
                                             padding)

                        if distp < best_dist:
                            best_y = yp
                            best_x = xp
                            best_dist = distp

                    i = i + 1
                    mag = rs_start * (alpha ** i)

                f[col, row] = np.array([best_y, best_x])
                dist[col, row] = best_dist

        save_idx = []
        hist.append(np.copy(f))
        print(f"Done {iter}")

    return f, dist, hist