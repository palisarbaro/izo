import cv2
import math
import  numpy as np
from lab2_remove_noise import medianFilter
target = 'p.jpg'
image = cv2.imread(f"./img/{target}")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
h,w = image.shape

DEBUG_MOD = False # set True for using fast numpy function instead of mine :)

def F(u,v,image):
    M = image.shape[0]
    N = image.shape[1]
    res = 0
    for x in range(M):
        for y in range(N):
            res += image[x][y] * math.e**(-1j*2*math.pi*(u*x/M + v*y/N))
    return res

def f(u,v, image):
    M = image.shape[0]
    N = image.shape[1]
    res = 0
    for x in range(M):
        for y in range(N):
            res += image[x][y] * math.e ** (1j * 2 * math.pi * (u * x / M + v * y / N))
    return res / M / N


def Fourier(image):
    if DEBUG_MOD:
        return np.fft.fft2(image)
    else:
        res = image.astype('complex')
        for u in range(image.shape[0]):
            for v in range(image.shape[1]):
                res[u][v] = F(u,v,image)
        return res

def rFourier(image):
    if DEBUG_MOD:
        return np.fft.ifft2(image)
    else:
        res = image.astype('complex')
        for u in range(image.shape[0]):
            for v in range(image.shape[1]):
                res[u][v] = f(u, v, image)
        return res


def test_my_Fourier(image):
    global DEBUG_MOD
    tmp = DEBUG_MOD
    DEBUG_MOD = False
    my_f = Fourier(image)
    DEBUG_MOD = True
    np_f = Fourier(image)
    res = np.abs(my_f-np_f)
    print(res.max())
    if res.max()<0.0001:
        print('Прямое работает')
    else:
        print('Прямое не работает')
    DEBUG_MOD = False
    my_r = rFourier(my_f)
    DEBUG_MOD = True
    np_r = rFourier(np_f)
    res = np.abs(my_r-np_r)
    print(res.max())
    if res.max()<0.0001:
        print('Обратное работает')
    else:
        print('Обратное не работает')

    DEBUG_MOD = tmp


def D(u,v,M,N):
    return ((M//2-u) ** 2 + (N//2-v) ** 2) ** 0.5


def getLowFilter(M,N,R):
    filter = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if D(i,j,M,N)<R:
                filter[i][j]=1
    return filter


def getHighFilter(M,N,R):
    return 1-getLowFilter(M,N,R)


def showFourier(image,name):
    image = np.abs(image)
    mean = image.mean()
    image *= 255/mean
    image = image.astype(np.uint8)
    cv2.imshow(name,image)

M = image.shape[0]
N = image.shape[1]

#test_my_Fourier(image)

original_image = image.astype(np.uint8)
noised_image = image.astype(np.uint8)
for i in range(M):
    for j in range(N):
        if (i)%6==0 and j%6==0:
            noised_image[i][j] = 0
fourier = Fourier(noised_image)
fourier_low = fourier*getLowFilter(M,N,100)
fourier_high = fourier*getHighFilter(M,N,100)
image_low = np.abs(rFourier(fourier_low)).astype(np.uint8)
image_high = np.abs(rFourier(fourier_high)).astype(np.uint8)
median = medianFilter(noised_image,3)

cv2.imshow("original", original_image)
cv2.imshow("noised", noised_image)
cv2.imshow("low", image_low)
cv2.imshow("high", image_high)
cv2.imshow("median",median)
showFourier(Fourier(original_image),"original F")
showFourier(fourier_low,"low f")
showFourier(fourier_high,"high f")

cv2.waitKey(0)
cv2.destroyAllWindows()