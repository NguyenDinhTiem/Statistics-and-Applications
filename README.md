# **Statistics and Applications**
#### Author: Nguyễn Đình Tiềm
## **INTRODUCTION**
Mùa thu này, trong khóa học AIFC mình học được mấy thứ hay ho như tính Mean, Median,Median, Range, Variance... mình gặp nó mấy lần trong toán rồi, giờ học lại thấy nó đơn giản cực , nhưng ứng dụng của nó thật hay ho, rất thú vị.

### **NỘI DUNG:** 

1. Mean and Median
2. Range and Histogram
3. Variance
4. Correlation Coefficient

# 1. Mean and Median
Mean là thuật toán tính trung bình giá trị các điểm ảnh, gọi X là tập các giá trị điểm ảnh, N là số các điểm ảnh khi đó trung bình giá trị các điểm ảnh bằng tổng giá trị các điểm ảnh chia số phần tử N.
![a.png](https://github.com/NguyenDinhTiem/Statistics-and-Applications/blob/main/c1.png)




Thuật toán Mean có thể ứng dụng dùng làm mờ các điểm ảnh, một vùng hay toàn bộ bức ảnh.

Ví dụ: Mình có 1 bức ảnh, mình muốn áp dụng thuật toán Mean để làm mờ khuôn mặt trong ảnh đó.
Trước tiên mình cần import numpy và import cv2, load ảnh lên,sau đó mình tạo một Kenel-cửa sổ có kích thước 5x5 rồi mình dùng cv2.filter2D để tính Mean cho tấm ảnh. Kết quả thu được sẽ là 1 bức ảnh mờ nhòe không rõ ràng.



```python
# load image and blurring

import numpy as np
import cv2

# load image in grayscale mode
image = cv2.imread('mrbean.jpg', 0)

# create kernel
kernel = np.ones((3,3), np.float32) / 9.0

# compute mean for each pixel
dst = cv2.filter2D(image, cv2.CV_8U, kernel)

# show images
cv2.imshow('image', image)
cv2.imshow('dst', dst)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Mình muốn làm mờ mỗi khuôn mặt thôi, vì vậy ý tưởng là lấy giá trị của mỗi phần khuôn mặt đem tính mean, sau đó lại gán lại vào bức ảnh thì mình sẽ có được bức ảnh mờ mỗi phần mặt


```python
# numpy review
import numpy as np

arr = np.ones((5,5))
print(arr)

roi = arr[1:4, 1:4]
roi = roi + 1
print(roi)

arr[1:4, 1:4] = roi
print(arr)
```

    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    [[2. 2. 2.]
     [2. 2. 2.]
     [2. 2. 2.]]
    [[1. 1. 1. 1. 1.]
     [1. 2. 2. 2. 1.]
     [1. 2. 2. 2. 1.]
     [1. 2. 2. 2. 1.]
     [1. 1. 1. 1. 1.]]
    

Code hoàn chỉnh, khác bên trên alf mình thêm roi là phần khuôn mặt và tính mean cho nó thôi


```python
# load image and blurring using mask-simple

import numpy as np
import cv2

# load image in grayscale mode
image = cv2.imread('mrbean.jpg', 0)

# create kernel
kernel = np.ones((5,5), np.float32) / 25.0

# Select ROI (top_y,top_x,height, width)
roi = image[40:140,150:280]

# compute mean for each pixel
roi = cv2.filter2D(roi, cv2.CV_8U, kernel)

image[40:140,150:280] = roi

# show images
cv2.imshow('roi', roi)
cv2.imshow('image', image)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Ngoài ra ở CV2 còn hỗ trợ hàm cv2.CascadeClassifier('haarcascade_frontalface_default.xml') phát hiện khuôn mặt rất tiện nên mình đỡ phải xác định roi ở phần code mình trình bày bên trên.


```python
# load image and blurring using face detection

import numpy as np
import cv2

# face detection setup
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load image in grayscale mode
image = cv2.imread('mrbean.jpg', 1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# face detection
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
# create kernel
kernel = np.ones((7,7), np.float32) / 49.0

# Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    roi = image[y:y+h,x:x+w]

    # compute mean for each pixel
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)

    # update
    image[y:y+h,x:x+w] = roi

# show images
cv2.imshow('image', image)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```

b) Thuật toán Mean có thể giải quyết tốt các bức ảnh không có nhiễu, khi xử lí các bức ảnh có nhiễu thì nó không còn mô tả đúng bức ảnh. Thuật toán Median có thể khác phục điều đó.
Trong một của sổ Kenel có kích thước m*m đang xét đặt của sổ này vào bức ảnh, lấy giá trị trong phạm vi cửa sổ đang xét rồi sắp xếp từ nhỏ đến lớn. Nếu m chẵn thì giá trị điểm ảnh = điểm ảnh có vị trị (m+1)/2 . Nếu m lẻ thì giá trị điểm ảnh = [phần tử thứ m/2 + phần tử thứ (m/2 + 1)]/2.
(Cách hiểu của mình có vẻ nhì nhằng nhưng các bạn xem công thức là hiểu liền :)))
VD: Số điểm ảnh là lẻ
![a.png](https://github.com/NguyenDinhTiem/Statistics-and-Applications/blob/main/md1.png)
vd: Số phần tử điểm ảnh là chẵn
![a.png](https://github.com/NguyenDinhTiem/Statistics-and-Applications/blob/main/md2.png)

VD: Xử lí nhiễu ảnh bằng Median


```python
import numpy as np
import cv2

img1 = cv2.imread('mrbean_noise.jpg')
img2 = cv2.medianBlur(img1, 9)

# show images
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 2. Range and Histogram
a) Range
Mình có 1 tập dữ liệu, Range của tập này  =  max - min :))
VD: 

```
def find_range(numbers):
    lowest = min(numbers)
    highest = max(numbers)
    r = highest - lowest
    print('Lớn nhất:{0}\t Nhỏ nhất:{1}\t Range:{2}'.format(lowest, highest, r))
#data
point=[1,4,6,3,5,4,3,44,6,5,4,66,88,2,3,1,9]
find_range(point)
```
b)Histogram
Histogram ứng dụng để cân bằng độ sáng trong ảnh, cái này mình học trong môn xử lí ảnh, mình dùng thuật toán kẻ bảng giải cũng không khó lắm nhưng mình chưa chuyển thành code được :)), hôm nào đẹp trời mình sẽ viết riêng phần này :)))


# 3.Variance 
Để tính variance thì trước hết ta cần tính Mean tập X, sau đó tính Variancle bằng cách lấy Tổng của hiệu từng phần tử trừ đi mean bình phương  sau đó chia cho số phần tử của tập đang xét.
![a.png](https://github.com/NguyenDinhTiem/Statistics-and-Applications/blob/main/v1.png)

Ứng dụng của variance để tìm texture, biên, cạnh trong ảnh

VD: Cài đặt thuật toán 

```
# variance
def calculate_mean(numbers):                    #1
    s = sum(numbers)
    N = len(numbers)
    mean = s/N
    return mean

def caculate_variance(numbers):                 #2
    mean = calculate_mean(numbers)              #3
    
    diff = []                                   #4
    for num in numbers: 
         diff.append(num-mean)
            
    squared_diff = []                           #5
    for d in diff:
        squared_diff.append(d**2)
    
    sum_squared_diff = sum(squared_diff)
    variance = sum_squared_diff/len(numbers)
        
    return variance

# data
points = [7, 8, 9, 2, 10, 9, 9, 9, 9, 4, 5, 6, 1, 5, 6, 7, 8, 6, 1, 10]

print('Phương sai của dãy số là: ', caculate_variance(points))
print('Độ lệch chuẩn của dãy số là: ', caculate_variance(points)**0.5)
```



VD: Xửa lí ảnh, tìm texture của ảnh.
Trong 1 bức ảnh các vùng ảnh có màu giống nhau thì có độ chênh lệch Variance rất nhỏ, những vùng ảnh có màu lệch khác nhau có độ chênh lệch variance lớn. Vì vậy dựa vào độ chênh lệch này ta có thể xác định biên, cachnj của các đối tượng trong ảnh.


```
import numpy as np
import cv2
import math
from scipy.ndimage.filters import generic_filter

img  = cv2.imread('mavel.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('mv1.jpg', gray)

x = gray.astype('float')
x_filt = generic_filter(x, np.std, size=7)
cv2.imwrite('mv2.jpg', x_filt)

x_filt[x_filt < 20] = 0
cv2.imwrite('mv3.jpg', x_filt)

maxv = np.max(x_filt)
print(maxv)

x_filt = x_filt*2.5
cv2.imwrite('mv4.jpg', x_filt)
```



# 4. Correlation Coefficient - Hệ Số Tương Quan
Hệ số tương quan là chỉ số thống kê đo lường mức độ mạnh yếu của mối quan hệ giữa hai biến số.
![a.png](https://github.com/NguyenDinhTiem/Statistics-and-Applications/blob/main/c1.png)
Chứng minh:
![a.png](https://github.com/NguyenDinhTiem/Statistics-and-Applications/blob/main/c2.png)

![a.png](https://github.com/NguyenDinhTiem/Statistics-and-Applications/blob/main/c3.png)

Cài đặt thuật toán:


```
def find_corr_x_y(x,y):                                         #1
    n = len(x)                                                  #2
    prod = []
    for xi,yi in zip(x,y):                                      #3
         prod.append(xi*yi)
         
    sum_prod_x_y = sum(prod)                                    #4
    
    sum_x = sum(x)
    sum_y = sum(y)
    
    squared_sum_x = sum_x**2
    squared_sum_y = sum_y**2 
    
    x_square = []
    for xi in x:
        x_square.append(xi**2)            
    x_square_sum = sum(x_square)
    y_square=[]
    for yi in y:
        y_square.append(yi**2)        
    y_square_sum = sum(y_square)
    
    # Use formula to calculate correlation                      #5
    numerator = n*sum_prod_x_y - sum_x*sum_y
    denominator_term1 = n*x_square_sum - squared_sum_x
    denominator_term2 = n*y_square_sum - squared_sum_y
    denominator = (denominator_term1*denominator_term2)**0.5
    correlation = numerator/denominator
    
    return correlation
```


```
x = [7, 18, 29, 2, 10, 9, 9]
y = [1, 6, 12, 8, 6, 21, 10]
y2 = []
for i in y:
    y2.append(i*2+1)
print(y2)
    
print('Hệ số tương quan tuyến tính giữa hai biến x,y: ',find_corr_x_y(x,y))
print('Hệ số tương quan tuyến tính giữa hai biến x,y: ',find_corr_x_y(x,y2))
```




Ứng dụng cho patch matching

![a.png](https://github.com/NguyenDinhTiem/Statistics-and-Applications/blob/main/c4.png)


```python
# patch matching
import numpy as np
from PIL import Image

# load ảnh và chuyển về kiểu list
image1 = Image.open('images/patch1.png')
image2 = Image.open('images/patch2.png')
image3 = Image.open('images/patch3.png')
image4 = Image.open('images/patch4.png')

image1_list = np.asarray(image1).flatten().tolist()
image2_list = np.asarray(image2).flatten().tolist()
image3_list = np.asarray(image3).flatten().tolist()
image4_list = np.asarray(image4).flatten().tolist()

# tính correlation coefficient
corr_1_2 = find_corr_x_y(image1_list, image2_list)
corr_1_3 = find_corr_x_y(image1_list, image3_list)
corr_1_4 = find_corr_x_y(image1_list, image4_list)

print('corr_1_2:', corr_1_2)
print('corr_1_3:', corr_1_3)
print('corr_1_4:', corr_1_4)
```

Bài viết dựa trên tài liệu khóa học AI Foundation Course in AI VietNam
