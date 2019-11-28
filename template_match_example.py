import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('currencysymbols.jpg', 0)
if len(img.shape) > 2:
    print("This image is in color..")
    print("Converting it into a grayscale image")
    grey_img = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)

template = cv.imread('dollarsymbol.jpg', 0)
if len(template.shape) > 2:
    print("This template image is in color..")
    print("Converting it into a grayscale template image")
    template = cv.cvtColor(src=template, code=cv.COLOR_BGR2GRAY)
else:
    print("Image is already in GrayScale mode")
    temp_w, temp_h = template.shape[::-1]
    img_w, img_h = img.shape[::-1]

img2 = img.copy()
print("Template width={0},height={1}".format(temp_w, temp_h))

if temp_w < img_w and temp_h < img_h:
    img = img2.copy()
    # method = eval('cv.TM_SQDIFF')
    # method = eval('cv.TM_CCOEFF_NORMED')

    # Apply template matching
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF)
    loc = np.where(res >= 0.9)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Take minimum since we are using TM_SQDIFF
    top_left = min_loc
    bottom_right = (top_left[0] + temp_w, top_left[1] + temp_h)
    cv.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=3)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv.waitKey(0)
else:
    print("Image height and width must be less than template width and height")
