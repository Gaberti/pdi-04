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
        
        blur = cv2.bilateralFilter(img, 11, 85, 101)

        # CannyAccThresh = cv2.threshold(img ,0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # CannyThresh = 0.1 * CannyAccThresh
        
        edges = cv2.Canny(blur, 75, 100)
        
        '''blur = cv2.bilateralFilter(img, 9, 75, 75)
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 2)
        
        #ret, threshold2 = cv2.threshold(threshold, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((3,3), np.float32)
        abertura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=3)
        fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel, iterations=1)'''


        cv2.imwrite(f'test{nome}.png', edges)
        

if __name__ == '__main__':
    main()