import cv2 as cv
import numpy as np

def caluculate_cdf(img_hist):
    cdf = img_hist.cumsum()
    cdf = cdf / float(np.max(cdf))
    return cdf

def caluculate_lookup(src_cdf, ref_cdf):
    lookup_table = np.zeros(256)
    pixelVal = 0

    for i in range(len(src_cdf)):
        pixelVal    
        for j in range(len(ref_cdf)):
            if ref_cdf[j] >= src_cdf[i]:
                pixelVal = j
                break
        lookup_table[i] = pixelVal
    return lookup_table



def match_histogram(src_img, ref_img):
    hist_src, _ = np.histogram(src_img.flatten(), 256, [0, 256])
    hist_ref, _ = np.histogram(ref_img.flatten(), 256, [0, 256])

    src_cdf, ref_cdf = caluculate_cdf(hist_src), caluculate_cdf(hist_ref)

    result =  cv.LUT(src_img, caluculate_lookup(src_cdf, ref_cdf))

    return cv.convertScaleAbs(result)


img = cv.imread('../../Week2/aspens_in_fall.jpg')
ref_img = cv.imread('../../Week2/forest-resized.jpg')

img_b, img_g, img_r = cv.split(img)
ref_img_b, ref_img_g, ref_img_r = cv.split(ref_img)


matched_image = cv.merge([match_histogram(img_b, ref_img_b), match_histogram(img_g, ref_img_g), match_histogram(img_r, ref_img_r)])
cv.imshow('Histogram Matched', matched_image)
cv.waitKey(0)


