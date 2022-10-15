import sys
import numpy as np
import cv2


INPUT_IMAGES = (
    ('60', '60.bmp'),
    ('82', '82.bmp'),
    ('114', '114.bmp'),
    ('150', '150.bmp'),
    ('205', '205.bmp')
)

def main():
    for nome, img in INPUT_IMAGES:
        img = cv2.imread(img, 0)
        if img is None:
            print('Failed to open image. \n')
            sys.exit()

        #img = img.astype (np.float32) / 255
        
        blur = cv2.bilateralFilter(img, d=-1, sigmaColor=50, sigmaSpace=3)
        
        #blur = cv2.adaptiveThreshold(img, maxValue=)

        adaptativeThreshold = cv2.adaptiveThreshold(blur, C=-5, blockSize=31, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY)

        val, globalThreshold = cv2.threshold(blur, thresh=0, maxval=1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        mask = adaptativeThreshold * globalThreshold

        #out = cv2.GaussianBlur(out, ksize=(0, 0), sigmaX=2.0)

        #out = blur[:] - (out[:] + 1)

        #out = (out.astype(np.float32)/255) * blur

        #out = np.where(out > 200, out, 0)
        #out = out*255
        #out = blur*0.5 + (out * blur)*0.5

        # CannyAccThresh = cv2.threshold(img ,0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # CannyThresh = 0.1 * CannyAccThresh
        
        #edges = cv2.Canny(out, 75, 100, 5)
        
        '''blur = cv2.bilateralFilter(img, 9, 75, 75)
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 2)
        
        #ret, threshold2 = cv2.threshold(threshold, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((3,3), np.float32)
        abertura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=3)
        fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel, iterations=1)'''


        cv2.imwrite(f'test{nome}.png', mask*255)
        

if __name__ == '__main__':
    main()