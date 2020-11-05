from lab2_remove_noise import medianFilter
from common_functions import *
import time

target = 'p.jpg'
image = cv2.imread(f"./img/{target}")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
t1 = time.perf_counter()
my_median = medianFilter(image, 1)
t2 = time.perf_counter()
cv_median = cv2.medianBlur(image,3)
t3 = time.perf_counter()
cv2.imshow("cv",cv_median)
cv2.imshow("my",my_median)
print(cv_median[33][33],my_median[33][33]) # Результаты работы одинаковые
print(f"cv time:{t3-t2},  my time:{t2-t1}")
cv2.waitKey(0)
cv2.destroyAllWindows()