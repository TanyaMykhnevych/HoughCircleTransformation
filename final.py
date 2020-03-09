from collections import defaultdict
import glob
import cv2
import numpy as np
from math import cos, sin, pi


def display_image(image):
    if image.shape[0] > 1280:
        scale_percent = 25
    else:
        scale_percent = 100

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_copy = cv2.resize(image, dim)
    cv2.imshow('image', image_copy)
    cv2.waitKey()


def sobel_filter(image):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    dx = cv2.filter2D(image, -1, kx)
    dy = cv2.filter2D(image, -1, ky)

    g = np.hypot(dx, dy)
    g = np.array(g / g.max() * 255, np.uint8)
    theta = np.arctan2(dy, dx)
    return g, theta


def non_max_suppression(image, theta):
    M, N = image.shape
    Z = np.zeros((M, N), dtype=np.uint8)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180
    print("nonmax")
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                    q = image[i, j + 1]
                    r = image[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = image[i + 1, j - 1]
                    r = image[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = image[i + 1, j]
                    r = image[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = image[i - 1, j - 1]
                    r = image[i + 1, j + 1]

                if image[i, j] >= q and image[i, j] >= r:
                    Z[i, j] = image[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass

    return np.array(Z, np.uint8)


def threshold(img, low_threshold_ratio=0.15, high_threshold_ratio=0.25):
    high_threshold_ratio = img.max() * high_threshold_ratio
    low_threshold_ratio = high_threshold_ratio * low_threshold_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)

    weak = np.uint8(125)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(img >= high_threshold_ratio)
    weak_i, weak_j = np.where((img < high_threshold_ratio) & (img >= low_threshold_ratio))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return np.array(res, np.uint8)


def binary_image(img):
    weak = 125
    strong = 255
    M, N = img.shape

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError:
                    pass
    return np.array(img, np.uint8)


def voting(edges, min_radius, max_radius):
    array = defaultdict(int)
    result_image = np.zeros((edges.shape[0], edges.shape[1]), np.uint8)
    blank_image = np.zeros((edges.shape[0], edges.shape[1]))
    for r in range(min_radius, max_radius):
        for i in range(0, edges.shape[0]):
            for j in range(0, edges.shape[1]):
                if edges[i][j] == 255:
                    for teta in range(0, 360, 5):
                        a = int(i - r * cos(teta * pi / 180))
                        b = int(j - r * sin(teta * pi / 180))
                        try:
                            blank_image[a][b] += 1
                            array[(a, b, r)] += 1
                        except IndexError:
                            continue
            print(r, i)

    maxim = np.amax(blank_image)

    for i in range(blank_image.shape[0]):
        for j in range(blank_image.shape[1]):
            result_image[i][j] = int(255 * float(blank_image[i][j] * 1.0 / maxim * 1.0))

    return result_image, array


def extract_circles(array):
    circles_list = []
    for k in sorted(array, key=lambda i: -i[2]):
        x, y, r = k
        if array[k] >= 16 and x > 0 and y > 0 and all(
                (x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles_list):
            print(array[k], x, y, r)
            circles_list.append((x, y, r))
    return circles_list


def draw_circles(circles_list, image):
    for circle in circles_list:
        cv2.circle(image, (circle[1], circle[0]), circle[2], (255, 0, 0), 2)
    #display_image(image)


def main(image_name, min_radius, max_radius, gaussian_kernel_dimension = 5, lowThresholdRatio = 0.15, highThresholdRatio = 0.25):
    image = cv2.imread(image_name)
    img_orig = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #display_image(image)

    image = cv2.GaussianBlur(image, (gaussian_kernel_dimension, gaussian_kernel_dimension), 0)
    #display_image(image)

    image, theta = sobel_filter(image)
    #display_image(image)

    image = non_max_suppression(image, theta)
    #display_image(image)

    image = threshold(image, lowThresholdRatio, highThresholdRatio)
    #display_image(image)

    image = binary_image(image)
    #display_image(image)

    image, array = voting(image, min_radius, max_radius)
    #display_image(image)

    circles = extract_circles(array)
    draw_circles(circles, img_orig)

    cv2.imwrite(image_name.replace("source", "output"), img_orig)

for filename in glob.glob(r"*"): 
    main(filename, 10, 20, 5, 0.3, 0.5)