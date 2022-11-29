import os
import cv2
import hdbscan
import numpy as np
from convert import getBMP, PNGtoBMP, BMPtoSVG
from typing import Dict, List
from numpy.typing import ArrayLike, NDArray


def binarize_image(image):
    blur = cv2.medianBlur(image, 9)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def get_letter_contours(thresh_image, area_low=100, area_high=10000):
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        parent = contours[hierarchy[0][i][3]] if hierarchy[0][i][3] > 0 else np.float32([[0, 0]])
        if area_high > area > area_low and not area_high > cv2.contourArea(parent) > area_low:
            x, y, w, h = cv2.boundingRect(cnt)
            if h / w > 0.1:
                bounding_boxes.append([x, y, w, h])

    return bounding_boxes


def group_contours(bounding_boxes, threshold=10000):
    letters = []
    for a in bounding_boxes:
        # Get the closest bounding boxes to a
        others = bounding_boxes.copy()
        boxes = list(filter(lambda b: (a[0] + a[2] / 2 - b[0] - b[2] / 2) ** 2 + (a[1] + a[3] / 2 - b[1] - b[3] / 2) ** 2 < threshold, others))

        # Get the bounding box
        x, y = min([a[0] for a in boxes]), min([a[1] for a in boxes]),
        xw, yh = max([a[0] + a[2] for a in boxes]), max([a[1] + a[3] for a in boxes])
        w, h = xw - x, yh - y

        if [x, y, w, h] not in letters:
            letters += [[x, y, w, h]]

    return letters


def get_letters_grid(letters, debug=False, debug_directory=None):
    # Sort these boxes by left to right, top to bottom
    x_coordinates = np.float32([a[1] for a in letters])

    # HDBSCAN
    x_coordinates = np.reshape(x_coordinates, (-1, 1))
    clustering = hdbscan.HDBSCAN().fit(x_coordinates)
    labels = clustering.labels_

    letters_grid = []
    for i in np.unique(labels):
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


def resize_img(img: NDArray, new_height: int = None, new_width: int = None):
    h, w, _ = img.shape

    if new_height is not None and new_width is not None:
        return cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    if new_height is not None:
        return cv2.resize(img, dsize=(int(new_height * w / h), new_height), interpolation=cv2.INTER_CUBIC)
    if new_width is not None:
        return cv2.resize(img, dsize=(new_width, int(new_width * h / w)), interpolation=cv2.INTER_CUBIC)
    else:
        return img


def add_vertical_blanks(img: NDArray, bottom: int, top: int) -> NDArray:
    _, w, _ = img.shape
    # Get background color
    color = img[0, 0, :]
    # Set top and bottom margins
    white_image_top = np.full((top, w, 3), color, dtype=np.uint8)
    white_image_bottom = np.full((bottom, w, 3), color, dtype=np.uint8)
    # Concatenate them around the image
    return np.concatenate((white_image_top, img, white_image_bottom), axis=0)


def convert_length(ref_height, target_width):
    # ref_height must correspond to 8/10 of the final glyph size
    return int(target_width * ref_height / 800)


def tweak_full_height(letters: Dict[str, NDArray]) -> Dict[str, NDArray]:
    characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "1234567890" + "!?/%&()[]"
    # Get max height of these characters
    max_height = max([letters[c].shape[0] for c in characters])
    # Resize characters to this size
    for character in characters:
        # h, w, _ = letters[character].shape
        # TODO: Note sure about this, this erases the way I unconsciously set letters heights when writing
        #letters[character] = resize_img(letters[character], new_height=max_height)

        # When letters are two wide, then sie them down
        #if letters[character].shape[1] > convert_length(max_height, 1000):
        #    letters[character] = resize_img(letters[character], new_width=convert_length(max_height, 1000))

        letters[character] = add_vertical_blanks(letters[character], convert_length(max_height, 200), max_height - letters[character].shape[0])

    return letters


def tweak_small_height(letters: Dict[str, NDArray]) -> Dict[str, NDArray]:
    characters: str = "labcdehikmnorstuvwxz.:"
    # Get max height of characters
    # TODO Store max_height from tweak_full_height rather than compute it again
    max_height = letters['A'].shape[0]
    # Either size down to max_height or add empty space on top of letter
    for character in characters:
        h, w, _ = letters[character].shape
        if h >= max_height:
            letters[character] = resize_img(letters[character], new_height=max_height)
        else:
            letters[character] = add_vertical_blanks(letters[character], convert_length(max_height, 200),
                                                     max_height - letters[character].shape[0])
    return letters


def tweak_under_baseline(letters: Dict[str, NDArray]) -> Dict[str, NDArray]:
    characters: str = "fgjpqy;"
    # Get max height of characters
    # TODO Store max_height from tweak_full_height rather than compute it again
    max_height = letters['A'].shape[0]

    for character in characters:
        h, w, _ = letters[character].shape
        if h >= max_height:
            letters[character] = resize_img(letters[character], new_height=max_height)
        else:
            letters[character] = add_vertical_blanks(letters[character], 0, max_height - letters[character].shape[0] + convert_length(max_height, 200))
    return letters


def tweak_apostrophes(letters: Dict[str, NDArray]) -> Dict[str, NDArray]:
    characters: str = "\'\""
    # Get max height of characters
    # TODO Store max_height from tweak_full_height rather than compute it again
    max_height = letters['A'].shape[0]

    for character in characters:
        h, w, _ = letters[character].shape
        if h >= max_height:
            letters[character] = resize_img(letters[character], new_height=max_height)
        else:
            letters[character] = add_vertical_blanks(letters[character], max_height - letters[character].shape[0] + convert_length(max_height, 200), 0)
    return letters


def tweak_signs(letters: Dict[str, NDArray]) -> Dict[str, NDArray]:
    characters: str = "-+="
    # Get max height of characters
    # TODO Store max_height from tweak_full_height rather than compute it again
    max_height = letters['A'].shape[0]

    for character in characters:
        h, w, _ = letters[character].shape
        if h >= max_height:
            letters[character] = resize_img(letters[character], new_height=max_height)
        else:
            # Get middle of "a"
            mid_height = letters['a'].shape[0] // 2 + convert_length(max_height, 200)
            bottom_space = convert_length(max_height, 200) + mid_height - letters[character].shape[0] // 2
            letters[character] = add_vertical_blanks(letters[character], bottom_space, convert_length(max_height, 1000) - bottom_space - letters[character].shape[0])

    return letters


class PhotoToGlyphs:

    source = 'photos/'
    debug_directory = 'debug/photo/'
    glyphs_directory = 'glyphs/'
    template = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?"\'-+=/%&()[]'

    def __init__(self, image_path, debug=False):
        self.debug_value = debug

        self.image = cv2.imread(PhotoToGlyphs.source + image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.binary = binarize_image(self.gray)
        self.debug('binary') if debug else None

        self.contours_boxes = get_letter_contours(self.binary)
        self.debug('contours') if debug else None

        self.letters_contours = group_contours(self.contours_boxes)
        self.debug('letters') if debug else None

        self.letters_grid = get_letters_grid(self.letters_contours)
        self.debug('letters_grid') if debug else None

        self.letters = self.get_letters()
        self.export_letters('png')

        PNGtoBMP(PhotoToGlyphs.glyphs_directory)
        self.letters = getBMP(PhotoToGlyphs.glyphs_directory)
        self.tweak_letters()
        self.export_letters('bmp')

        BMPtoSVG(PhotoToGlyphs.glyphs_directory)

    def debug(self, which='all'):
        os.makedirs(PhotoToGlyphs.debug_directory, exist_ok=True)

        if self.debug_value and which in ['all', 'binary']:
            # Save binary image
            cv2.imwrite(PhotoToGlyphs.debug_directory + 'binary.png', self.binary)

        if self.debug_value and which in ['all', 'contours']:
            # Save first contours boxes drawn on photo
            image = self.image.copy()
            for (x, y, w, h) in self.contours_boxes:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imwrite(PhotoToGlyphs.debug_directory + 'contours_boxes.png', image)

        if self.debug_value and which in ['all', 'letters']:
            # Save refined letter contours
            image = self.image.copy()
            for (x, y, w, h) in self.letters_contours:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(PhotoToGlyphs.debug_directory + 'letters_contours.png', image)

        if self.debug_value and which in ['all', 'letters_grid']:
            all_letters = []
            for i, (x, y, w, h) in iterate_grid(self.letters_grid):
                letter_img = self.image[y:y + h, x:x + w]
                all_letters.append(letter_img)

            # Resize letters
            max_height = max([a.shape[1] for a in all_letters])
            all_letters = [cv2.resize(img, dsize=(int(max_height * img.shape[1] / img.shape[0]), max_height), interpolation=cv2.INTER_CUBIC) for img in all_letters]

            new_im = np.concatenate(all_letters, axis=1)
            cv2.imwrite(PhotoToGlyphs.debug_directory + 'letters_grid.png', new_im)

    def get_letters(self) -> Dict[str, NDArray]:
        letters: Dict[str, NDArray] = dict()
        for i, (x, y, w, h) in iterate_grid(self.letters_grid):
            letter_img: NDArray = self.image[y:y + h, x:x + w]
            letters[PhotoToGlyphs.template[i]] = letter_img
        return letters

    def export_letters(self, extension):
        os.makedirs(PhotoToGlyphs.glyphs_directory, exist_ok=True)
        for character, letter_img in self.letters.items():
            glyph = str(ord(character))
            # Create directory for this glyph
            os.makedirs(PhotoToGlyphs.glyphs_directory + glyph, exist_ok=True)
            # Saving the letter
            cv2.imwrite(PhotoToGlyphs.glyphs_directory + glyph + f'/{glyph}.{extension}', letter_img)

    def tweak_letters(self):
        self.letters = tweak_full_height(self.letters)
        self.letters = tweak_signs(self.letters)
        self.letters = tweak_small_height(self.letters)
        self.letters = tweak_under_baseline(self.letters)
        self.letters = tweak_apostrophes(self.letters)


if __name__ == '__main__':
    for file in os.listdir(PhotoToGlyphs.source):
        PhotoToGlyphs(file, debug=True)
