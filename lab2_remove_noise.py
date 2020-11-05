from random import gauss
from common_functions import *
def medianFilter(image, eps):
    g = np.copy(image)
    h,w = g.shape
    for i in range(h):
        for j in range(w):
            around = []
            for di in range(-eps,eps+1):
                for dj in range(-eps,eps+1):
                    y = i + di
                    x = j + dj
                    if x>=0 and x<w and y>=0 and y<h:
                        around.append(image[y][x])
            around.sort()
            g[i][j] = around[len(around)//2]
    return g

def G(x,y,sigma):
    return 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))

def gaussFilter(image,eps,sigma):
    g = np.copy(image)
    h, w = g.shape
    matrix = {(x,y): G(x,y,sigma) for x in range(-eps,eps+1) for y in range(-eps,eps+1)}
    summ = sum(matrix.values())
    for pos in matrix:
        matrix[pos] /= summ
    for i in range(h):
        for j in range(w):
            s = 0
            for di in range(-eps, eps + 1):
                for dj in range(-eps, eps + 1):
                    y = i + di
                    x = j + dj
                    if x >= 0 and x < w and y >= 0 and y < h:
                        s+=image[y][x]*matrix[(di,dj)]
            g[i][j] = max(min(s,255),0)
    return g


if __name__== '__main__':
    target = 'p.jpg'
    image = cv2.imread(f"./img/{target}")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    median = medianFilter(image,1)
    gauss = gaussFilter(image,1,50)
    show_gisto(image, "image_g")
    show_gisto(median,"median_g")
    show_gisto(gauss,"gauss_g")
    cv2.imshow("image",image)
    cv2.imshow("median",median)
    cv2.imshow("gauss",gauss)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
