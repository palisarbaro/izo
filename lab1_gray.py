import cv2
target = 'trees.jpg'
image = cv2.imread(f"./img/{target}")
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()