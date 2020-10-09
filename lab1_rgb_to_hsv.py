import cv2
import numpy as np

def img_rgb_to_hsv(image):
    image2 = np.copy(image)
    h, w, _ = image.shape
    for i in range(h):
        for j in range(w):
            b,g,r = image[i][j]
            image2[i][j] = rgb_to_hsv(r,g,b)
    return image2


def rgb_to_hsv(r,g,b):
    r /= 255.
    g /= 255.
    b /= 255.
    Min = min([r,g,b])
    Max = max([r,g,b])
    V = Max
    if V!=0:
        S = (V - Min)/V
    else:
        S = 0
    d = Max - Min
    if d==0:
        H = 0
    elif V==r:
        H = 60*(g-b)/d
    elif V==g:
        H = 120+60*(b-r)/d
    elif V==b:
        H = 240+60*(r-g)/d
    if H<0:
        H+=360
    H/=2
    return (int(H),int(S*255),int(V*255))

def img_hsv_to_rgb(image):
    image2 = np.copy(image)
    h, w, _ = image.shape
    for i in range(h):
        for j in range(w):
            H,S,V = image[i][j]
            image2[i][j] = hsv_to_rgb(H,S,V)[::-1]
    return image2

def hsv_to_rgb(H,S,V):
    H *= 2
    Hi = int(H / 60.) % 6
    Vmin = (255. - S) * V / 255.
    a = (V - Vmin) * (H % 60) / 60
    Vinc = Vmin + a
    Vdec = V - a
    return [
        (V, Vinc, Vmin),
        (Vdec, V, Vmin),
        (Vmin, V, Vinc),
        (Vmin, Vdec, V),
        (Vinc, Vmin, V),
        (V, Vmin, Vdec)
    ][Hi]

def increase_brightness(image):
    image2 = np.copy(image)
    h, w, _ = image.shape
    for i in range(h):
        for j in range(w):
            H,S,V = rgb_to_hsv(*image[i][j][::-1])
            V = min(255,V+50)
            image2[i][j] = hsv_to_rgb(H,S,V)[::-1]
    return image2


target = 'trees.jpg'
image = cv2.imread(f"./img/{target}")


print("#a")
print(image[50][50])
image2 = img_rgb_to_hsv(image)
print(image2[50][50])
image2 = img_hsv_to_rgb(image2)
print(image2[50][50])


print("#b")
print(image[50][50])
image2 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
print(image2[50][50])
image2 = cv2.cvtColor(image2,cv2.COLOR_HSV2BGR)
print(image2[50][50])

print("#d")
image2 = increase_brightness(image)
cv2.imshow("Image1", image)
cv2.imshow("Image2", image2)

from lab1_ssim import ssim
print("#e")
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print(ssim(image,image2))


cv2.waitKey(0)
cv2.destroyAllWindows()