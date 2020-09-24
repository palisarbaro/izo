import cv2
target = 'trees.jpg'
image = cv2.imread(f"./img/{target}")
image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
h,w,_ = image.shape
for i in range(h):
    for j in range(w):
        b,g,r = image[i][j]
        image[i][j] = 0.2952*r + 0.5547*g + 0.148*b
print(image[500][50],image2[500][50]) # результат немного разный, т.к.  в алгоритме cvtColor немого другие коэффициенты
cv2.imshow("Image1", image)
cv2.imshow("Image2", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()