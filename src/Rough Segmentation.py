import os
import shutil
import SKFCM as skfcm
import cv2 as cv
import imghdr
import pydicom
import numpy as np
from numba import jit
import random as rng

L0, L1, R0, R1 = 0, 0, 0, 0
classes = 5
flag = False


# @jit
def read_img(loc):
    img_type = imghdr.what(loc)
    if img_type is None:
        ds = pydicom.read_file(loc)

        rows = ds.get(0x00280010).value
        cols = ds.get(0x00280011).value

        # instance_number = int(ds.get(0x00200013).value)
        try:
            window_center = int(ds.get(0x00281050).value[0])
            window_width = int(ds.get(0x00281051).value[0])
        except TypeError:
            window_center = int(ds.get(0x00281050).value)
            window_width = int(ds.get(0x00281051).value)

        window_max = int(window_center + window_width / 2)
        window_min = int(window_center - window_width / 2)

        if ds.get(0x00281052) is None:
            rescale_intercept = 0
        else:
            rescale_intercept = int(ds.get(0x00281052).value)

        if ds.get(0x00281053) is None:
            rescale_slope = 1
        else:
            rescale_slope = int(ds.get(0x00281053).value)

        img = np.zeros((rows, cols), np.uint8)
        pixels = ds.pixel_array

        for i in range(0, rows):
            for j in range(0, cols):
                pix_val = pixels[i][j]
                rescale_pix_val = pix_val * rescale_slope + rescale_intercept

                if rescale_pix_val > window_max:
                    img[i][j] = 255
                elif rescale_pix_val < window_min:
                    img[i][j] = 0
                else:
                    img[i][j] = int(((rescale_pix_val - window_min) / (window_max - window_min)) * 255)
    else:
        img = cv.imread(loc, 0)
    return img


@jit
def in_range(x, y, img):
    return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]


@jit
def preprocess(img):
    h, w = img.shape
    print([h, w])
    for x in range(h):
        for y in range(w):
            count_up, count_down, count_left, count_right = 0, 0, 0, 0
            for i in range(1, 8):
                if in_range(x+i, y, img) and img[x+i][y] <= 5:
                    count_up += 1
                if in_range(x-i, y, img) and img[x-i][y] <= 5:
                    count_down += 1
            for j in range(1, 4):
                if in_range(x, y+j, img) and img[x][y+j] <= 5:
                    count_left += 1
                if in_range(x, y-j, img) and img[x][y-j] <= 5:
                    count_right += 1
            if count_up and count_down:
                img[x][y] = 0
            if count_right and count_left:
                img[x][y] = 0
    img = cv.medianBlur(img, 3)
    return img

# def find_cavity(img):
#     x0, y0 = int(img.shape[0]/2), int(img.shape[1]/2)
#     left, right, up, down = -1, -1, -1, -1
#     i = 1
#     while i < x0 and (up < 0 or down < 0):
#         x_u, x_d = x0 - i, x0 + i
#         if up < 0 and check_bond(img, x_u, y0, -2):
#             up = x_u
#         if down < 0 and check_bond(img, x_d, y0, 2):
#             down = x_d
#         i += 1
#
#     x0 = int(up + (down-up)/2)
#     i = 1
#     while i < y0 and (left < 0 or right < 0):
#         y_l, y_r = y0 - i, y0 + i
#         if left < 0 and check_bond(img, x0, y_l, -1):
#             left = y_l
#         if right < 0 and check_bond(img, x0, y_r, 1):
#             right = y_r
#         i += 1
#
#     center = int(up + (down-up)/2 + (down - up)/10)
#     cv.circle(img, (left, x0), 5, 255, -1)
#     cv.circle(img, (right, x0), 5, 255, -1)
#     cv.circle(img, (y0, up), 5, 255, -1)
#     cv.circle(img, (y0, down), 5, 255, -1)
#     return left, right, up, down, center
# def check_bond(img, x, y, dire):
#     step = 10
#     count = 0
#     if dire == -1:
#         for i in range(0, step):
#             if (in_range(x, y-i, img) and img[x][y-i] <= 5) or not in_range(x, y-i, img):
#                 count += 1
#     elif dire == 1:
#         for i in range(0, step):
#             if (in_range(x, y+i, img) and img[x][y+i] <= 5) or not in_range(x, y+i, img):
#                 count += 1
#     elif dire == -2:
#         for i in range(0, step):
#             if (in_range(x-i, y, img) and img[x-i][y] <= 5) or not in_range(x-i, y, img):
#                 count += 1
#     elif dire == 2:
#         for i in range(0, step):
#             if (in_range(x+i, y, img) and img[x+i][y] <= 5) or not in_range(x+i, y, img):
#                 count += 1
#
#     return count == step


def find_cavity_ellipse(img, thresh=0):
    _, binary = cv.threshold(img, thresh, 255, 0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    # cv.imshow("binary", binary)
    # cv.waitKey(10)

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    size = 0
    ((cx, cy), (a, b), angle) = ((0, 0), (0, 0), 0)
    for c in range(len(contours)):
        if contours[c].shape[0] <= 5:
            continue
        if contours[c].shape[0] > size:
            ((cx, cy), (a, b), angle) = cv.fitEllipse(contours[c])
            size = contours[c].shape[0]
    # cv.ellipse(img, (np.int32(cx), np.int32(cy)),
    #            (np.int32(a / 2), np.int32(b / 2)), angle, 0, 360, (255, 255, 255), 2, 8, 0)
    #
    # cv.imshow("contour", img)
    # cv.waitKey(10)

    return (cx, cy), (a, b), angle


# def crop_rect(img):
#     (cx, cy), (l2, l1), angle = find_cavity_ellipse(img, 55)
#     x0, y0 = cy + 0.06*l2, cx
#     cv.circle(img, (int(y0), int(x0)), 5, 0, -1)
#     cv.imshow("p", img)
#     cv.waitKey(0)
#
#     # left
#     y = y0 - l1/4
#     start = (int(y - l1/8), int(x0 - l1/8))
#     end = (int(y + l1/8 + l1/20), int(x0 + l1/8))
#     img = cv.rectangle(img, start, end, (0, 0, 0), 5)
#
#     # right
#     y = y0 + l1 / 4
#     start = (int(y - l1 / 8 - l1/20), int(x0 - l1 / 8))
#     end = (int(y + l1 / 8), int(x0 + l1 / 8))
#     img = cv.rectangle(img, start, end, (0, 0, 0), 5)
#
#     cv.imshow("rect", img)
#     cv.waitKey(0)


def candidate_kidney_region(src):
    center, (l2, l1), angle = find_cavity_ellipse(src, 55)
    src_gray = np.array(src)
    X = (center[0], center[1] + 0.06*l2)
    # cv.circle(src_gray, (int(X[0]), int(X[1])), 25, 255)
    y0, x0, x1 = X[1], X[0] - 0.28*l1, X[0] + 0.28*l1
    region1 = ((int(x0), int(y0)), (0.4*l1, 0.6*l2), 30)
    region2 = ((int(x1), int(y0)), (0.4*l1, 0.6*l2), -30)
    # cv.ellipse(src_gray, region1, 255, 2)
    # cv.ellipse(src_gray, region2, 255, 2)
    # cv.imshow("candidate region", src_gray)
    # cv.waitKey(10)
    mask = np.zeros_like(src_gray)
    cv.ellipse(mask, region1, color=(255, 255, 255), thickness=-1)
    cv.ellipse(mask, region2, color=(255, 255, 255), thickness=-1)
    extracted_region = np.bitwise_and(src, mask)
    # cv.imshow("region", extracted_region)
    # cv.waitKey(10)
    return extracted_region, region1, region2, X


def canny(img):
    t = 80
    canny_output = cv.Canny(img, t, t*2)
    return canny_output


def get_kidney(img_origin, img, side, first, last=False, X=(0, 0), rects=None, previous=None):
    global classes, flag
    if last:
        min_px = 300
    elif first:
        min_px = 2500
    else:
        min_px = 1500

    if first:
        extracted_region, left, right, X = candidate_kidney_region(img)
        region_mask = np.zeros_like(img)

        # Note the CT scans are mirrored, so the left kidney of the patient should be in the right side of the image
        if side == 'left':
            cv.ellipse(region_mask, right, color=(255, 255, 255), thickness=-1)
            extracted_region = cv.bitwise_and(extracted_region, region_mask)
            # cv.imshow("img", extracted_region)
            # cv.waitKey(0)
        else:
            cv.ellipse(region_mask, left, color=(255, 255, 255), thickness=-1)
            extracted_region = cv.bitwise_and(extracted_region, region_mask)

        exclusive, V = skfcm.SKFCM(extracted_region, sigma=100, n_classes=classes, m=2, alpha=4, itr_times=500, epsilon=1e-7, shared=False)
    else:
        rects = rects.astype(np.int)
        x, y, h, w = rects[1][1], rects[1][0], rects[1][3], rects[1][2]
        # left_region, right_region = np.zeros_like(img), np.zeros_like(img)
        # cv.rectangle(left_region, (y1, x1), (y1+w1, x1+h1),
        #              (255, 255, 255), -1)
        # cv.rectangle(right_region, (y2, x2), (y2 + w2, x2 + h2),
        #              (255, 255, 255), -1)
        exclusive, V = skfcm.SKFCM(img, sigma=100, n_classes=classes, m=2, alpha=4, itr_times=500,
                                   epsilon=1e-7, first=False, shared=True, x0=x, y0=y, h=h, w=w)

    thresh = V[classes-2]
    # cv.imshow("exclusive_result", exclusive)
    # cv.waitKey(0)
    exclusive[exclusive <= thresh] = 0
    exclusive[exclusive > thresh] = 255
    binary = exclusive.astype(np.uint8)
    exclusive_res = np.array(binary)
    if first:
        # cv.imshow("binary", binary)
        # cv.waitKey(10)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
        # cv.imshow("open", binary)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    elif not flag:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    binary, labels, stats, kidney_label = locate_kidney(binary, min_px, last, X)

    if first and kidney_label == -1:
        binary, labels, stats, kidney_label = locate_kidney(exclusive_res, 2000, last, X)
        if kidney_label == -1:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            binary = cv.morphologyEx(exclusive_res, cv.MORPH_CLOSE, kernel)
            binary, labels, stats, kidney_label = locate_kidney(binary, 1500, last, X)
        if kidney_label == -1:
            raise NameError('need to decrease num_classes')

    _, binary = cv.threshold(binary, 0, 255, 0)
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    binary = cv.dilate(binary, kernel1)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel2)

    # cv.imshow("kidney", kidney)
    # cv.waitKey(0)

    # fit rectangle
    binary, rects = fit_rectangle(binary, first, last, rects, kidney_label, previous)
    kidney = cv.bitwise_and(img_origin, binary)

    #     cv.rectangle(kidney, (x, y), (x + w, y + h),
    #                  (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)), 5)
    # cv.imshow("rect", kidney)
    # cv.waitKey(10)
    # cv.destroyAllWindows()

    return kidney, binary, rects, X


def locate_kidney(binary, min_px, last, X):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, 4, cv.CV_32S)
    regions = []
    for i in range(0, num_labels):
        if stats[i][4] <= min_px or (last and stats[i][4] >= 500):
            binary[labels == i] = 0
        elif stats[i][4] <= 100000:
            regions.append(i)

    dist_to_spine = 100000
    kidney_label = -1
    for label in regions:
        if dist_to_spine > dist(X, centroids[label]):
            kidney_label = label

    for label in regions:
        if label != kidney_label:
            binary[labels == label] = 0
    return binary, labels, stats, kidney_label


def fit_rectangle(binary, first, last, rects, kidney_label, previous=None):
    global flag
    if kidney_label == -1:
        flag = True
        if previous is not None:
            if last:
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
                previous = cv.erode(previous, kernel)
            binary = previous

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(contours[0])
    if first:
        rects[0] = [x, y, w, h]
    else:
        if abs(x - rects[1][0]) >= rects[1][2] / 2 or abs(y - rects[1][1]) >= rects[1][3] / 2 or \
                w - rects[1][2] >= rects[1][2] / 6 or h - rects[1][3] >= rects[1][3] / 6 or \
                rects[1][2] - w >= rects[1][2] / 3 or rects[1][3] - h >= rects[1][3] / 3:
            if previous is not None:
                if last:
                    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
                    previous = cv.erode(previous, kernel)
                binary = previous
                contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv.boundingRect(contours[0])
                if last:
                    x, y, w, h = x - 10, y - 10, w + 20, h + 20

    rects[1] = [x, y, w, h]

    #     cv.rectangle(kidney, (x, y), (x + w, y + h),
    #                  (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)), 5)
    # cv.imshow("rect", kidney)
    # cv.waitKey(10)
    # cv.destroyAllWindows()
    return binary, rects


def dist(p1, p2):
    return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2))


def process_slice(prefix, dest, side, start_num, num, rects, X, last=False, binary=None):
    img_origin = read_img(prefix + str(int(num)) + ".dcm")
    img = cv.medianBlur(img_origin, 3)
    img = preprocess(img)
    if start_num == num:
        kidney, binary, rects, X = get_kidney(img_origin, img, side, first=True, X=X, last=last, rects=rects)
    else:
        kidney, binary, rects, _ = get_kidney(img_origin, img, side, False, last, X, rects, binary)

    res = cv.imread(dest+"/"+str(num)+".jpg", 0)
    if res is None:
        cv.imwrite(dest+"/"+str(num)+".jpg", kidney)
    else:
        res += kidney
        cv.imwrite(dest+"/"+str(num)+".jpg", res)
    return binary, rects


def segment(prefix, dest, side):
    global classes, L0, L1, R0, R1, flag
    if side == 'left':
        start, end = L0, L1
    else:
        start, end = R0, R1
    start_num = int((start + end) / 2)
    X = (0, 0)
    rects = np.zeros((2, 4))
    binary = None
    first_binary = None

    try:
        for num in range(start_num, end + 1):
            last = abs(num - end) <= abs(end - start) * 0.08
            try:
                binary, rects = process_slice(prefix, dest, side, start_num, num, rects, X, last, binary)
            except NameError:
                classes -= 1
                binary, rects = process_slice(prefix, dest, side, start_num, num, rects, X, last, binary)
            if num == start_num:
                first_binary = binary
    except:
        start_num -= 4
        for num in range(start_num, end + 1):
            last = abs(num - end) <= abs(end - start) * 0.08
            binary, rects = process_slice(prefix, dest, side, start_num, num, rects, X, last, binary)
            if num == start_num:
                first_binary = binary

    rects[1] = rects[0]
    binary = first_binary
    for num in range(start_num - 1, start - 1, -1):
        last = abs(num - start) <= abs(start - end) * 0.08
        binary, rects = process_slice(prefix, dest, side, start_num, num, rects, X, last, binary)


def rough_segment(id, seq, series, startL, endL, startR, endR, n_classes):
    global L0, L1, R0, R1, classes, flag
    flag = False
    classes = n_classes
    L0, L1, R0, R1 = startL, endL, startR, endR
    prefix = "seattlechildrens/" + id + "/" + str(id) + ".Seq" + str(seq) + ".Ser" + str(series) + ".Img"
    dest = str(id) + "_kidney"
    shutil.rmtree(dest)
    os.makedirs(dest)
    segment(prefix, dest, "left")
    classes = n_classes
    flag = False
    segment(prefix, dest, "right")

    # start_num = int((endL + startL + endR + startR) / 4)
    # X = (0, 0)
    # rects = np.zeros((4, 4))
    # binary = None
    # first_binary = None
    #
    # # left:
    # for num in range(start_num, min(endL, endR) + 1):
    #     last = abs(num-min(endL, endR)) <= abs(max(startR, startL)-min(endL, endR)) * 0.08
    #     binary, rects = process_slice(prefix, dest, start_num, num, rects, n_classes, X, last, binary)
    #     if num == start_num:
    #         first_binary = binary
    #
    # rects[2], rects[3] = rects[0], rects[1]
    # binary = first_binary
    # for num in range(start_num-1, max(startL, startR) - 1, -1):
    #     last = abs(num-max(startR, startL)) <= abs(max(startR, startL)-min(endL, endR)) * 0.08
    #     binary, rects = process_slice(prefix, dest, start_num, num, rects, n_classes, X, last, binary)


def main():
    rough_segment("23", 4, 4, 43, 83, 64, 92, 5)


    # candidate_kidney_region(img)
    # crop_rect(img)

    # extracted_region, left, right, mask = candidate_kidney_region(img)
    # exclusive, V = skfcm.SKFCM(extracted_region, sigma=100, n_classes=4, m=2, alpha=4, itr_times=500, epsilon=1e-7)
    # exclusive[exclusive < V[3]] = 0
    # exclusive = exclusive / 255
    # # cv.imwrite("example result.jpg", exclusive*255)
    # cv.imshow("exclusive_result", exclusive)
    # cv.waitKey(0)

    # exclusive = skfcm.SKFCM(img, sigma=100, n_classes=4, m=2, alpha=4, itr_times=500, epsilon=1e-7)
    # exclusive = preprocess(exclusive)
    # exclusive = exclusive / 255

    # # cv.imwrite("experiment.jpg", exclusive*255)
    # cv.imshow("exclusive_result", exclusive)
    # cv.waitKey(0)


if __name__ == "__main__":
    main()
