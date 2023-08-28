import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image

# 이미지 출력 메소드
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
# 데이터셋 불러오기
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# 각 변수에 110번째 이미지 정보 저장
img = x_train[109]
label = t_train[109]
print(label)

print(img.shape)
# numpy의 reshape 함수를 이용하여 28 X 28의 형태로 변환
img = img.reshape(28, 28)
print(img.shape)

img_show(img)