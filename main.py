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

        
        blur = cv2.bilateralFilter(img, d=-1, sigmaColor=50, sigmaSpace=3)
        
        adaptativeThreshold = cv2.adaptiveThreshold(blur, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=101, C=-27)

        otsuThreshold, globalThreshold = cv2.threshold(blur, thresh=0, maxval=1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        mask = adaptativeThreshold * globalThreshold

        kernel = np.ones((3,3), np.float32)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

        blobs = []

        for label in range(1, n):
            blobs.append(stats[label, cv2.CC_STAT_AREA])

        blobs.sort()

        median = np.median(blobs)
        
        blobs = [round(blob/median) if blob > median else 1 for blob in blobs]

        print(sum(blobs))
        
        
        cv2.imwrite(f'test{nome}.png', mask*255)

if __name__ == '__main__':
    main()