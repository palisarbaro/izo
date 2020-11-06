import cv2
import numpy as np
from common_functions import *
from random import random
from math import sqrt

def gauss(mu,sigma):
    N = 50
    return sigma*(sqrt(12./N)*sum([random()-0.5 for i in range(N)])) + mu

def addGauss(image,mu,sigma):
    g = np.copy(image)
    h,w = g.shape
    for i in range(h):
        for j in range(w):
            noise = gauss(mu,sigma)
            res = max(min(int(g[i][j])+noise,255),0)
            g[i][j] = res

    return g

if __name__== '__main__':
    target = 'trees.jpg'
    image = cv2.imread(f"./img/{target}")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    show_gisto(image,"image_g")
    Gauss = addGauss(image,10,5)
    show_gisto(Gauss,"gauss_g")
    cv2.imshow("gauss",Gauss)
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
