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
        
        blur = cv2.medianBlur(img, 5)
        #blur2 = cv2.GaussianBlur(blur, (5,5), 0)
        #img = img.astype(np.float32)/255
        
        threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 2)
        #ret, threshold2 = cv2.threshold(threshold, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((1,1), np.uint8)
        #sure_bg = cv2.dilate(threshold, kernel,iterations=1)
        abertura = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=3)
        fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel, iterations=3)
        #img3 = cv2.subtract(sure_bg, sure_fg)
        sure_fg = cv2.erode(fechamento, kernel, iterations=3)


        cv2.imwrite(f'test{nome}.png', blur)
        

if __name__ == '__main__':
    main()