import sys
import numpy as np
import cv2


INPUT_IMAGES = (
    ('60', '60.bmp'),
    ('82', '82.bmp'),
    ('114', '114.bmp'),
    ('150', '150.bmp'),
    ('205', '205.bmp'),
)

def main():
    for nome, img in INPUT_IMAGES:
        img = cv2.imread(img, 0)
        if img is None:
            print('Failed to open image. \n')
            sys.exit()

        #img = img.astype (np.float32) / 255
        
        blur = cv2.bilateralFilter(img, d=-1, sigmaColor=50, sigmaSpace=3)
        
        adaptativeThreshold = cv2.adaptiveThreshold(blur, C=-5, blockSize=33, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY)

        otsuThreshold, globalThreshold = cv2.threshold(blur, thresh=0, maxval=1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        mask = adaptativeThreshold * globalThreshold

        kernel = np.ones((3,3), np.float32)
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        #mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=1)
        #mask_dilated = cv2.morphologyEx(mask_close, cv2.MORPH_DILATE, kernel, iterations=1)

        #masked_img = img * mask_dilated
        #canny = cv2.Canny(blur, threshold1=0.5*otsuThreshold, threshold2=otsuThreshold, apertureSize=3)

        #contours, hierarchy = cv2.findContours((hsv_oriented[..., 2]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(bgr_oriented, contours, -1, (0, 255, 0), 1)
        
        cv2.imwrite(f'test{nome}.png', mask_open*255)
        

if __name__ == '__main__':
    main()