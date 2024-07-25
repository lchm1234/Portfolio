import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from Classification import MyModel, classes

# 모델 로드
model = MyModel(num_classes=len(classes))
state_dict = torch.load('model10epoch_64batch.pth')
model.load_state_dict(state_dict)
model.eval()

# # 모델을 ONNX로 변환
# dummy_input = torch.randn(1, 3, 224, 224)

# # 모델을 ONNX 파일로 내보냅니다.
# torch.onnx.export(model, dummy_input, "model10epoch_64batch.onnx")

# ONNX 모델을 TensorFlow로 변환
onnx_model = onnx.load('model10epoch_64batch.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('model.pb')

# TensorFlow 모델을 TensorFlow Lite로 변환
converter = tf.lite.TFLiteConverter.from_saved_model('model.pb')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)