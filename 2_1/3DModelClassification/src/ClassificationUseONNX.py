import onnxruntime
import numpy as np
from torchvision import transforms
from PIL import Image

# ONNX 모델 불러오기
onnx_model_path = 'model10epoch_64batch.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),          # 이미지를 Tensor로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
])

image_path = 'TEST_IRP.jpg'
image = Image.open(image_path)  # 이미지 열기
image = transform(image)        # 변환 적용
image_np = image.numpy()        # Tensor를 NumPy 배열로 변환

image_np = np.expand_dims(image_np, axis=0)  # 배치 차원 추가 (1, C, H, W)

# ONNX 모델을 통해 예측 수행
ort_inputs = {ort_session.get_inputs()[0].name: image_np}
ort_outs = ort_session.run(None, ort_inputs)

# 출력에서 확률이 가장 높은 클래스 선택
predicted_class = np.argmax(ort_outs[0])

print(f'Predicted class index: {predicted_class}')