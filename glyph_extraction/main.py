import cv2
from pathlib import Path
from svgtrace import trace

img = cv2.imread('photos/bgn.jpg')
img_gray = cv2.imread('photos/bgn.jpg', 0)


def binarization(img):
    img = cv2.medianBlur(img, 9)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return th3


def contours(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        parent = contours[hierarchy[0][i][3]] if hierarchy[0][i][3] > 0 else []
        if 10000 > area > 100 and not 10000 > cv2.contourArea(parent) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x, y, w, h])
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return bounding_boxes


bounding_boxes = contours(binarization(img_gray))

visual = img.copy()

threshold = 10000

letters = []
for a in bounding_boxes:
    # Get the closest bounding boxes to a
    others = bounding_boxes.copy()
    boxes = list(filter(lambda b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < threshold, others))
    bound = [min([a[0] for a in boxes]),
             min([a[1] for a in boxes]),
             max([a[0] + a[2] for a in boxes]),
             max([a[1] + a[3] for a in boxes])]

    # Draw the bounding box
    x, y, xw, yh = bound
    w, h = xw - x, yh - y
    visual = cv2.rectangle(visual, (x, y), (x + w, y + h), (0, 0, 255), 2)

    letters += [[x, y, w, h]]


cv2.imwrite('result_binarization.jpg', visual)

# Extract letters from the image
for letter in letters:
    x, y, w, h = letter
    letter_img = img[y:y + h, x:x + w]
    cv2.imwrite(f'tiles/{x}_{y}.png', letter_img)
