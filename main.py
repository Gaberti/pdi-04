import sys
import numpy as np
import cv2


INPUT_IMAGES = (
#    ('60', '60.bmp'),
#    ('82', '82.bmp'),
#    ('114', '114.bmp'),
#    ('150', '150.bmp'),
    ('205', '205.bmp'),
)

def orientation_filter(hsv, janela, value_thresh, angle_thresh, mask):
    h_jan, w_jan = (v//2 for v in janela)
    height, width, c = hsv.shape
    out = hsv.copy()
    for y in range(h_jan, height - h_jan):
        for x in range(w_jan, width - w_jan):
            center = hsv[y, x]
            #centerAngle = (center[0] + 90) % 360
            if center[2] < value_thresh:
                continue
            if mask[y, x] == 0:
                continue
            #window = [(-1, -1),  (-1, 1),
            #          ( 0, -1),  ( 0, 1),
            #          ( 1, -1),  ( 1, 1),]
            #if (centerAngle > 45 and centerAngle < 135) or (centerAngle > 225 and centerAngle < 315):
            #    window = [(-1, -1), (-1, 0), (-1, 1),
            #              ( 1, -1), ( 1, 0), ( 1, 1),]
            #for dy, dx in window: 
            for dy in range(-h_jan, h_jan + 1):
                for dx in range(-w_jan, w_jan + 1):
                    if dy == dx == 0 or mask[y + dy, x + dx] == 0 or hsv[y + dy, x + dx][2] > center[2]:
                        continue
                    diff = hsv[y + dy, x + dx][0] - center[0]
                    diff = abs((diff + 360 + 180) % 360 - 180)
                    if diff < angle_thresh:
                        out[y + dy, x + dx][2] = max(out[y + dy, x + dx][2], ((angle_thresh - diff)/angle_thresh)*center[2]) #*(hsv[y + dy, x + dx][2]*1.5)
                        #out[y + dy, x + dx][2] = min(out[y + dy, x + dx][2], center[2], 1.0)
                        #m = max(out[y + dy, x + dx][2], center[2])
                        #out[y + dy, x + dx][2] = m #center[2] #min(out[y + dy, x + dx][2]*1.5, 1.0)
                        #out[y, x][2] = m
                    #soma += img2[y + dy, x + dx]
            #img[y, x] = soma/area_janela
    return out

#def orientation_filter2(hsv, janela, value_thresh, angle_thresh, mask):
#    h_jan, w_jan = (v//2 for v in janela)
#    height, width, c = hsv.shape
#    out = hsv.copy()
#    for y in range(h_jan, height - h_jan):
#        for x in range(w_jan, width - w_jan):
#            center = hsv[y, x]
#            if center[2] < value_thresh:
#                continue
#            if mask[y, x] == 0:
#                continue
#            #maxNeighbor = None
#            minValue = 2.0
#            for dy in range(-h_jan, h_jan + 1):
#                for dx in range(-w_jan, w_jan + 1):
#                    if dy == dx == 0 or mask[y + dy, x + dx] == 0:
#                        continue
#                    value = out[y + dy, x + dx][2]
#                    if value > minValue or value < center[2]:
#                        continue
#                    diff = hsv[y + dy, x + dx][0] - center[0]
#                    diff = abs((diff + 360 + 180) % 360 - 180)
#                    if diff < angle_thresh:
#                        minValue = value
#                        #maxNeighbor, maxValue = (dy, dx), value
#            #if maxNeighbor:
#            out[y, x][2] = minValue if minValue < 1.0 else center[2] #out[y + maxNeighbor[0], x + maxNeighbor[1]][2] = max(maxValue, center[2])
#    return out
#
#def orientation_filter3(hsv, janela, value_thresh, angle_thresh, mask):
#    h_jan, w_jan = (v//2 for v in janela)
#    height, width, c = hsv.shape
#    out = hsv.copy()
#    for y in range(h_jan, height - h_jan):
#        for x in range(w_jan, width - w_jan):
#            center = hsv[y, x]
#            if center[2] < value_thresh:
#                continue
#            if mask[y, x] == 0:
#                continue
#            centerAngle = (center[0]) % 180 # + 90) % 180
#            window = [(0, -1), (0, 1)]
#            if (centerAngle >= 157.5):      window = window
#            elif (centerAngle >= 112.5):    window = [(-1, -1), (1, 1)]
#            elif (centerAngle >= 67.5):     window = [(1, 0), (-1, 0)]
#            elif (centerAngle >= 22.5):     window = [(1, -1), (-1, 1)]
#            #window = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
#            maxNeighbor = None
#            maxValue = 0
#            #for dy in range(-h_jan, h_jan + 1):
#                #for dx in range(-w_jan, w_jan + 1):
#            for dy, dx in window:
#                if dy == dx == 0 or mask[y + dy, x + dx] == 0 or hsv[y + dy, x + dx][2] > center[2]:
#                    continue
#                diff = hsv[y + dy, x + dx][0] - center[0]
#                diff = abs((diff + 360 + 180) % 360 - 180)
#                value = out[y + dy, x + dx][2]
#                if diff < angle_thresh and value > maxValue and value < center[2]:
#                    maxNeighbor, maxValue = (dy, dx), value
#            if maxNeighbor:
#                out[y + maxNeighbor[0], x + maxNeighbor[1]][2] = center[2] #max(maxValue, center[2])
#    return out

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

        #canny = cv2.Canny(blur, threshold1=0.5*otsuThreshold, threshold2=otsuThreshold, apertureSize=3)

        mask = adaptativeThreshold * globalThreshold

        kernel = np.ones((3,3), np.float32)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_dilated = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_dilated = cv2.morphologyEx(mask_dilated, cv2.MORPH_DILATE, kernel, iterations=1)

        #canny = canny * mask
        masked_img = img * mask_dilated

        #img1 = (img - np.min(img, initial=255, where=img!=0))/(np.max(img) - np.min(img, initial=255, where=img!=0))#np.ptp(img)
        min = np.min(masked_img, initial=255, where=masked_img!=0)
        masked_img = np.where(masked_img == 0, min, masked_img)

        masked_img = cv2.normalize(masked_img, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #canny = cv2.Canny(img, threshold1=0, threshold2=255, apertureSize=3)

        #normalized = img[:]
        #print(min)
        #normalized = cv2.normalize(img, normalized, 0, 255, norm_type=cv2.NORM_MINMAX)
        #cv2.normalize(img1, img, alpha=1, beta=255)

        #equalized = cv2.equalizeHist(img)

        dx = cv2.Sobel(masked_img, cv2.CV_64F, 1, 0, ksize=1)
        dy = cv2.Sobel(masked_img, cv2.CV_64F, 0, 1, ksize=1)

        mag = cv2.magnitude(dx, dy)
        orien = cv2.phase(dx, dy, angleInDegrees=True)
        #orien = orien / 2. # Go from 0:360 to 0:180 
        hsv = np.zeros((orien.shape[0], orien.shape[1], 3), dtype=np.float32)
        hsv[..., 0] = orien #orien*2 # H (in Opencv2 between 0:180)
        hsv[..., 1] = 1.0 #255 # S
        hsv[..., 2] = cv2.normalize(mag, None, 0, 1.0, cv2.NORM_MINMAX) # V 0:255

        hsv_oriented = hsv[:]
        for i in range(10):
            hsv_oriented = orientation_filter(hsv_oriented, (3, 3), 0, 30, mask_dilated)
            #hsv_oriented = orientation_filter(hsv_oriented, (3, 3), 0.5, 180, mask_dilated)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr_oriented = cv2.cvtColor(hsv_oriented, cv2.COLOR_HSV2BGR)

        #laplacian = cv2.Laplacian(equalized, cv2.CV_64F)

        #canny = cv2.Canny(equalized, threshold1=0.5*otsuThreshold, threshold2=otsuThreshold, apertureSize=3)

        #out = cv2.GaussianBlur(out, ksize=(0, 0), sigmaX=2.0)

        #out = blur[:] - (out[:] + 1)

        #out = (out.astype(np.float32)/255) * blur

        #out = np.where(out > 200, out, 0)
        #out = out*255
        #out = blur*0.5 + (out * blur)*0.5

        # CannyAccThresh = cv2.threshold(img ,0,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # CannyThresh = 0.1 * CannyAccThresh
        
        #edges = cv2.Canny(out, 75, 100, 5)

        contours, hierarchy = cv2.findContours((hsv_oriented[..., 2]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(bgr_oriented, contours, -1, (0, 255, 0), 1)
        
        '''blur = cv2.bilateralFilter(img, 9, 75, 75)
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 2)
        
        #ret, threshold2 = cv2.threshold(threshold, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((3,3), np.float32)
        abertura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=3)
        fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel, iterations=1)'''


        cv2.imwrite(f'test{nome}.png', bgr*255)
        cv2.imwrite(f'test{nome}-oriented.png', bgr_oriented*255)
        cv2.imwrite(f'test{nome}-diff.png', (bgr_oriented - bgr)*255)
        cv2.imwrite(f'test{nome}-mag.png', hsv_oriented[..., 2]*255)
        cv2.imwrite(f'test{nome}-countour.png', hsv_oriented[..., 2]*255)
        

if __name__ == '__main__':
    main()