import os
import cv2
import hdbscan
import numpy as np


def binarize_image(image):
    blur = cv2.medianBlur(image, 9)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def get_letter_contours(thresh_image, area_low=100, area_high=10000):
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        parent = contours[hierarchy[0][i][3]] if hierarchy[0][i][3] > 0 else []
        if area_high > area > area_low and not area_high > cv2.contourArea(parent) > area_low:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x, y, w, h])

    return bounding_boxes


def group_contours(bounding_boxes, threshold=10000):
    letters = []
    for a in bounding_boxes:
        # Get the closest bounding boxes to a
        others = bounding_boxes.copy()
        boxes = list(filter(lambda b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < threshold, others))

        # Get the bounding box
        x, y = min([a[0] for a in boxes]), min([a[1] for a in boxes]),
        xw, yh = max([a[0] + a[2] for a in boxes]), max([a[1] + a[3] for a in boxes])
        w, h = xw - x, yh - y


        if [x, y, w, h] not in letters:
            letters += [[x, y, w, h]]

    return letters


def get_letters_grid(letters):
    # Sort these boxes by left to right, top to bottom
    x_coordinates = np.float32([a[1] for a in letters])

    # HDBSCAN
    x_coordinates = np.reshape(x_coordinates, (-1, 1))
    clustering = hdbscan.HDBSCAN().fit(x_coordinates)
    labels = clustering.labels_

    letters_grid = []
    for i in range(labels.max()):
        row = [a for j, a in enumerate(letters) if labels[j] == i]
        row = sorted(row, key=lambda x: x[0])
        letters_grid.append(row)

    return sorted(letters_grid, key=lambda x: x[0][1])


def iterate_grid(grid):
    i = 0
    for row in grid:
        for element in row:
            yield i, element
            i += 1


class PhotoToGlyphs:

    source = 'photos/'
    debug_directory = 'debug/photo/'
    glyphs_directory = 'glyphs/'

    def __init__(self, image_path, template, debug=False):
        self.template = template
        self.image = cv2.imread(PhotoToGlyphs.source + image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.binary = binarize_image(self.gray)
        self.contours_boxes = get_letter_contours(self.binary)
        self.letters_contours = group_contours(self.contours_boxes)
        self.letters_grid = get_letters_grid(self.letters_contours)
        self.save_letters()

        if debug:
            self.debug()

    def debug(self):
        os.makedirs(PhotoToGlyphs.debug_directory, exist_ok=True)

        # Save binary image
        cv2.imwrite(PhotoToGlyphs.debug_directory + 'binary.png', self.binary)

        # Save first contours boxes drawn on photo
        image = self.image.copy()
        for (x, y, w, h) in self.contours_boxes:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imwrite(PhotoToGlyphs.debug_directory + 'contours_boxes.png', image)

        # Save refined letter contours
        image = self.image.copy()
        for (x, y, w, h) in self.letters_contours:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(PhotoToGlyphs.debug_directory + 'letters_contours.png', image)

    def save_letters(self):
        os.makedirs(PhotoToGlyphs.glyphs_directory, exist_ok=True)

        repetition = dict()
        for i, (x, y, w, h) in iterate_grid(self.letters_grid):
            letter_img = self.image[y:y + h, x:x + w]
            # Create directory for this glyph
            glyph = str(ord(self.template[i]))
            os.makedirs(PhotoToGlyphs.glyphs_directory + glyph, exist_ok=True)
            # Saving the letter
            repetition[glyph] = repetition.get(glyph, 0) + 1
            # TODO Enable repetition!
            # cv2.imwrite(Photo.glyphs_directory + glyph + f'/{glyph}_{repetition[glyph]}.png', letter_img)
            cv2.imwrite(PhotoToGlyphs.glyphs_directory + glyph + f'/{glyph}.png', letter_img)


if __name__ == '__main__':
    for file in os.listdir(PhotoToGlyphs.source):
        template = 'ABCDEFGHIJKLMNOABCDEFGHIJKLMNOABCDEFGHIJKLMNO' + 'PQRSTUVWXYZPQRSTUVWXYZPQRSTUVWXYZ' + 'abcdefghijklmnopqrabcdefghijklmnopqrstabcdefghijklmnopqrustuvwxyz'
        PhotoToGlyphs(file, template, debug=True)
