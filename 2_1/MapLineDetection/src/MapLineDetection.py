import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 읽어옵니다.
image = cv2.imread('map_005.jpg')

# 이미지 중앙에서 좌, 우, 상, 하 각각 30씩 잘라냅니다.
center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
cropped_image = image[center_y-50:center_y+50, center_x-50:center_x+50]

# 흰색 도로
lower_color1 = np.array([222, 222, 222])
upper_color1 = np.array([232, 232, 232])
mask1 = cv2.inRange(cropped_image, lower_color1, upper_color1)

# 연노란색 도로
lower_color2 = np.array([180, 210, 210])
upper_color2 = np.array([206, 222, 222])
mask2 = cv2.inRange(cropped_image, lower_color2, upper_color2)

# 노란색 도로
lower_color3 = np.array([120, 200, 210])
upper_color3 = np.array([160, 225, 225])
mask3 = cv2.inRange(cropped_image, lower_color3, upper_color3)

# 두 개의 마스크를 합칩니다.
combined_mask = cv2.bitwise_or(mask1, mask2)
# 두 개의 마스크를 합칩니다.
combined_mask = cv2.bitwise_or(mask1, mask3)

# 캐니 에지 검출기를 적용합니다.
edges = cv2.Canny(combined_mask, 100, 150)

# 색상 필터가 적용된 이미지를 화면에 표시합니다.
plt.subplot(131)
plt.imshow(combined_mask, cmap='gray')
plt.title('Color Filtered Image')
plt.axis('off')
threshold = 100

# 새로운 이미지 생성 (검은 배경)
result_image = np.zeros_like(edges)

# 색상 필터가 적용된 이미지에 대해 Hough 변환을 수행합니다.
while threshold > 0:
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=threshold)
    # 선이 검출되었는지 확인하고 그려줍니다.
    if lines is not None:
        # 찾은 라인들을 원본 이미지에 그립니다.
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        break
    else:
        threshold -= 1
        
# Lines에서 각 라인의 각도를 추출
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) + 90
else:
    print("No lines were detected.")
    angle = 0

print(angle)

# 결과 이미지를 화면에 표시합니다.
plt.subplot(132)
plt.imshow(result_image)
plt.title('Detected Lines')
plt.axis('off')

# 원본 이미지를 화면에 표시합니다.
plt.subplot(133)
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.title('Original Image (Cropped)')
plt.axis('off')

plt.show()