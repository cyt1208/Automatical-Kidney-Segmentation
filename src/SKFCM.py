import cv2 as cv
import numpy as np
import math
import random
from numba import jit


@jit
def kernel(x, vi, sigma):
    return math.exp(-(math.pow(x-vi, 2) / math.pow(sigma, 2)))


@jit
def in_range(x, y, img):
    return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]


@jit
def neighbor_mean(x, y, img):
    num = 0
    sum_val = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if in_range(x+dx, y+dy, img):
                num += 1
                sum_val += img[x+dx][y+dy]
    return sum_val/num


def init_U(img, n_classes):
    U = np.zeros((img.shape[0], img.shape[1], n_classes))
    for i in range(0, n_classes):
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                ind = random.randint(0, n_classes-1)
                if i == ind:
                    U[x][y][i] = 1
    return U


def init_V(n_classes):
    V = np.zeros(n_classes)
    V *= 255/n_classes
    return V


@jit
def updateV(img, U, V, n_classes, m, sigma, alpha, first=True, x0=0, y0=0, h=0, w=0):
    if first:
        h = img.shape[0]
        w = img.shape[1]
    for i in range(0, n_classes):
        num, denum = 0, 0
        for x in range(x0, x0+h):
            for y in range(y0, y0+w):
                val = img[x][y]
                val_mean = neighbor_mean(x, y, img)
                k = kernel(val, V[i], sigma)
                k_mean = kernel(val_mean, V[i], sigma)
                num += math.pow(U[x][y][i], m) * (k*val + alpha*k_mean*val_mean)
                denum += math.pow(U[x][y][i], m) * (k + alpha*k_mean)
        V[i] = num / denum


@jit
def updateU(img, U, V, n_classes, m, sigma, alpha, first=True, x0=0, y0=0, h=0, w=0):
    if first:
        h = img.shape[0]
        w = img.shape[1]
    for i in range(0, n_classes):
        for x in range(x0, x0+h):
            for y in range(y0, y0+w):
                val = img[x][y]
                val_mean = neighbor_mean(x, y, img)
                k = kernel(val, V[i], sigma)
                k_mean = kernel(val_mean, V[i], sigma)
                num = math.pow((1-k)+alpha*(1-k_mean), -1/(m-1))
                U[x][y][i] = num

    for x in range(x0, x0 + h):
        for y in range(y0, y0 + w):
            denum = 0
            for j in range(0, n_classes):
                denum += U[x][y][j]
            for j in range(0, n_classes):
                U[x][y][j] /= denum


@jit
def get_final_image(img, U, V, n_classes, first=True, shared=False, x0=0, y0=0, h=0, w=0):
    final_image = np.zeros((img.shape[0], img.shape[1]))
    V_sort = np.copy(V)
    V_sort = np.sort(V_sort)
    if first:
        h = img.shape[0]
        w = img.shape[1]

    for x in list(range(x0, x0 + h)):
        for y in list(range(y0, y0 + w)):
            if shared:
                value = 0
                for i in range(0, n_classes):
                    value += V[i] * U[x][y][i]
                final_image[x][y] = value
            else:
                class_num, class_val = 0, 0
                for i in range(0, n_classes):
                    if U[x][y][i] > class_val:
                        class_val = U[x][y][i]
                        class_num = i
                # if np.argwhere(V_sort == V[class_num])[0] <= 1:
                #     final_image[x][y] = 0
                # else:
                #     final_image[x][y] = V[class_num]
                final_image[x][y] = V[class_num]
    return final_image, V_sort


def SKFCM(img, sigma, n_classes, m, alpha, itr_times, epsilon, first=True, shared=False, x0=0, y0=0, h=0, w=0):
    U = init_U(img, n_classes)
    V = init_V(n_classes)

    for itr in range(0, itr_times):
        old_U = np.copy(U)

        updateV(img, U, V, n_classes, m, sigma, alpha, first, x0, y0, h, w)

        updateU(img, U, V, n_classes, m, sigma, alpha, first, x0, y0, h, w)

        print("iteration " + str(itr))
        print(str(np.max(np.abs(old_U - U))))

        if np.max(np.abs(old_U - U)) <= epsilon:
            break
    res, V = get_final_image(img, U, V, n_classes, first, shared, x0, y0, h, w)

    return res, V
