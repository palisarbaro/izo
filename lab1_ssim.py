import cv2
from common_functions import *
from random import randint,random


def intensity(image):
    res = 0
    h, w = image.shape
    for row in image:
        for val in row:
            res += val
    return res/(w*h)

def contrast(image):
    res = 0
    intens = intensity(image)
    h, w = image.shape
    for row in image:
        for val in row:
            res += (intens-val)**2
    return np.sqrt(res / (w * h - 1))

def covariance(image1, image2):
    h, w = image1.shape
    res = 0
    for i in range(h):
        for j in range(w):
            res+= int(image1[i][j])*int(image2[i][j])
    return res/(w*h) - intensity(image1)*intensity(image2)

def ssim(image1,image2):
    m1 = intensity(image1)
    m2 = intensity(image2)
    s1 = contrast(image1)
    s2 = contrast(image2)
    cov = covariance(image1, image2)
    return (2*m1*m2+ 0.0001)/(m1**2 + m2**2 + 0.0001) * (2*s1*s2 + 0.0001)/(s1**2+s2**2+0.0001) * (cov +0.0001)/(s1*s2+ 0.0001)

if __name__== '__main__':
    target = 'smile.jpg'
    image = cv2.imread(f"./img/{target}")
    image1,image2,image3 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w, _ = image.shape
    for x in range(w):
        for y in range(h):
            if random()<0.5:
                image2[x][y] = randint(0,255)
                image3[x][y] = 255 - image3[x][y]
    print(ssim(image1,image1))
    print(ssim(image1,image2))
    print(ssim(image1,image3))
    show_gisto(image1,"g1")
    show_gisto(image2,"g2")
    show_gisto(image3,"g3")
    cv2.waitKey(0)
    cv2.destroyAllWindows()