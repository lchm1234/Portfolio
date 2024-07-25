from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
import logging

app = Flask(__name__)

# ONNX 모델 로드
model_path = 'vgg16model10epoch_64batch.onnx'
ort_session = ort.InferenceSession(model_path)

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# POST 요청 처리
@app.route('/api/infer', methods=['POST'])
def infer():
    try:
        # 이미지 데이터 받기
        image_data_base64 = request.json.get('image')
        if image_data_base64 is None:
            return jsonify({'error': 'No image data provided'}), 400

        # Base64 디코딩 후 이미지로 변환
        try:
            image_data = base64.b64decode(image_data_base64)
        except Exception as e:
            logging.error(f"Base64 decode error: {e}")
            return jsonify({'error': 'Base64 decode error'}), 400

        # 디코딩된 이미지를 파일로 저장하여 확인
        try:
            with open("received_image.png", "wb") as f:
                f.write(image_data)
            logging.info("Image successfully saved as received_image.png")
        except Exception as e:
            logging.error(f"File write error: {e}")
            return jsonify({'error': 'File write error'}), 500

        try:
            image = Image.open(BytesIO(image_data))
            # RGBA 이미지를 RGB로 변환
            if image.mode == 'RGBA':
                image = image.convert('RGB')
        except Exception as e:
            logging.error(f"Image open error: {e}")
            return jsonify({'error': 'Image open error'}), 400

        # 이미지 변환 및 전처리
        image = transform(image)
        image = image.unsqueeze(0).numpy()  # 배치 차원 추가 및 numpy 배열로 변환

        # ONNX 추론 수행
        ort_inputs = {ort_session.get_inputs()[0].name: image}
        ort_outs = ort_session.run(None, ort_inputs)

        # 추론 결과 반환
        predicted_class = np.argmax(ort_outs[0], axis=1).item()

        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=5003)