import tensorflow as tf
from tensorflow.keras import backend as K

num_classes = 3

def iou_metric(y_true, y_pred):
    y_true = K.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = K.cast(y_pred > 0.5, 'float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
    iou = intersection / (sum_ - intersection + K.epsilon())
    return iou

def f1_score(y_true, y_pred):
    y_true = K.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_pred = K.cast(y_pred > 0.5, 'float32')
    tp = K.sum(y_true * y_pred, axis=[1, 2, 3])
    fp = K.sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    fn = K.sum(y_true * (1 - y_pred), axis=[1, 2, 3])

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

# Load the Keras model
keras_model = tf.keras.models.load_model('unet_model_keras.h5', custom_objects={'iou_metric': iou_metric, 'f1_score': f1_score})

# Convert the Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# 사용자 정의 함수 등록
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('unet_model.tflite', 'wb') as f:
    f.write(tflite_model)