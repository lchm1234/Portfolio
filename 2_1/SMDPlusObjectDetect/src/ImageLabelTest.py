import cv2

# 이미지 불러오기
image = cv2.imread('3.jpg')

# .txt 파일 열기
with open('3.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    # 각 줄을 공백으로 분리하여 숫자로 변환
    class_index, p1, p2, p3, p4 = map(float, line.split())
    
    p1 = p1 * 1920
    p2 = p2 * 1080
    p3 = p3 * 1920
    p4 = p4 * 1080
    # 좌상단과 우하단의 좌표 계산
    top_left = (int(p1 - (p3 / 2)), int(p2 + (p4 / 2)))
    bottom_right = (int(p1 + (p3 / 2)), int(p2 - (p4 / 2)))

    print(top_left)
    print(bottom_right)
    # 사각형 그리기
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# 이미지 보여주기
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()