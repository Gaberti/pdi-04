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

        adaptativeThreshold = cv2.adaptiveThreshold(blur, C=-5, blockSize=33, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY)

        otsuThreshold, globalThreshold = cv2.threshold(blur, thresh=0, maxval=1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        canny = cv2.Canny(blur, threshold1=0.5*otsuThreshold, threshold2=otsuThreshold, apertureSize=3)

        mask = adaptativeThreshold * globalThreshold

        kernel = np.ones((3,3), np.float32)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        #canny = canny * mask
        img = img * mask

        #img1 = (img - np.min(img, initial=255, where=img!=0))/(np.max(img) - np.min(img, initial=255, where=img!=0))#np.ptp(img)
        min = np.min(img, initial=255, where=img!=0)
        img = np.where(img == 0, min, img)
        equalized = cv2.equalizeHist(img)

        #dx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=5)
        #dy = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=5)

        #mag = cv2.magnitude(dx, dy)
        #orien = cv2.phase(dx, dy, angleInDegrees=True)
        #orien = orien / 2. # Go from 0:360 to 0:180 
        #hsv = np.zeros((orien.shape[0], orien.shape[1], 3), dtype=np.uint8)
        #hsv[..., 0] = orien # H (in Opencv2 between 0:180)
        #hsv[..., 1] = 255 # S
        #hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # V 0:255

        #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        #laplacian = cv2.Laplacian(equalized, cv2.CV_64F)

        #canny = cv2.Canny(equalized, threshold1=0.5*otsuThreshold, threshold2=otsuThreshold, apertureSize=3)

        #normalized = img[:]
        #print(min)
        #normalized = cv2.normalize(img, normalized, 0, 255, norm_type=cv2.NORM_MINMAX)
        #cv2.normalize(img1, img, alpha=1, beta=255)

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


        cv2.imwrite(f'test{nome}.png', equalized)
        

if __name__ == '__main__':
    main()