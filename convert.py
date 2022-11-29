import os
import cv2
import numpy as np
import subprocess


def getBMP(directory):
    output = dict()
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        path = f'{directory}/{filename}/'
        img = cv2.imread(f'{path}/{filename}.bmp')
        output[chr(int(filename))] = img
    return output


def PNGtoBMP(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        path = f'{directory}/{filename}/'

        img = cv2.imread(f'{path}/{filename}.png')
        thresh_glyph = threshold_glyph(img, 2)

        cv2.imwrite(f'{path}/{filename}.bmp', thresh_glyph)


def BMPtoSVG(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        path = f'{directory}/{filename}/'
        # Trace image
        subprocess.run(["potrace", "--svg", f'{path}/{filename}.bmp', '-o', f'{path}/{filename}.svg'])


def threshold_glyph(image, k):
    i = np.float32(image).reshape(-1, 3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(i, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img


if __name__ == "__main__":
    extract_glyphs()
