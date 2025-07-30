import cv2
import numpy as np
 
def calculate_cdf(histogram):
    cdf = histogram.cumsum()
    normalized_cdf = cdf / float(cdf.max())
    return normalized_cdf
 
def calculate_lookup(src_cdf, ref_cdf):
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table
 
def match_histograms(src_image, ref_image):
    src_hist, _ = np.histogram(src_image.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_image.flatten(), 256, [0, 256])

    src_cdf = calculate_cdf(src_hist)
    ref_cdf = calculate_cdf(ref_hist)

    lookUP_TABLE = calculate_lookup(src_cdf, ref_cdf)
 
    result = cv2.LUT(src_image, lookUP_TABLE)

    return cv2.convertScaleAbs(result)

img = cv2.imread('Week2/aspens_in_fall.jpg')
ref_img = cv2.imread('Week2/forest-resized.jpg')
cv2.imshow('Orig_Img', img)
cv2.imshow('Ref_img', ref_img)

img_b, img_g, img_r = cv2.split(img)
ref_img_b, ref_img_g, ref_img_r = cv2.split(ref_img)

matched_image = cv2.merge([match_histograms(img_b, ref_img_b), match_histograms(img_g, ref_img_g), match_histograms(img_r, ref_img_r)])
cv2.imshow('Histogram Matched', matched_image)
cv2.waitKey(0)