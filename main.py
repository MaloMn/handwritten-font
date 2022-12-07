from skimage.morphology import skeletonize
from skimage import img_as_uint, io
from skimage import color
from skimage.util import invert
import cv2

if __name__ == "__main__":
    img = cv2.imread('test.png', 0)
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('binarized.png', th)


    image = cv2.imread("binarized.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = color.rgb2gray(image)

    # perform skeletonization and invert image
    skeleton = skeletonize(invert(image))

    io.imsave('skeleton.png', img_as_uint(skeleton))
