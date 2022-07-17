import os
import cv2
import numpy as np
import subprocess


def quantimage(image, k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret, label, center = cv2.kmeans(i, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img


directory = os.fsencode('tiles')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img = cv2.imread(f'tiles/{filename}')

    q_img = quantimage(img, 2)
    filename = filename.replace('.png', '')
    cv2.imwrite(f'bitmap/{filename}.bmp', q_img)

    # Trace image
    subprocess.run(["potrace", "--svg", f'bitmap/{filename}.bmp', '-o', f'glyphs/{filename}.svg'])
