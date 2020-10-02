import cv2
import numpy as np
def get_gisto(image):
    res = np.zeros((256,),np.uint64)
    for row in image:
        for val in row:
            res[val] += 1
    return res/max(res)*256

def show_gisto(gisto,name="gisto"):
    if type(gisto[0])==np.ndarray:
        gisto = get_gisto(gisto)
    scale = 2
    image = np.zeros((256*scale,256*scale,3),np.uint8)
    for x in range(len(gisto)):
        g = (256-int(gisto[x]))*scale
        image = cv2.rectangle(image,(x*scale,g),((x+1)*scale,256*scale),(60,60,x),-1)
    cv2.imshow(name, image)