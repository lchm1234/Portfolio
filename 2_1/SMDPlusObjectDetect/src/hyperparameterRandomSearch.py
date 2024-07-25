from sklearn.model_selection import ParameterSampler
from ultralytics import YOLO
import numpy as np


def random_search(data, param_dist, n_iter):
    best_model = None
    best_score = float('-inf')
    best_params = None

    for params in ParameterSampler(param_dist, n_iter=n_iter):
        model = YOLO('yolov8n.pt')
        results = model.train(data=data, **params)

        score = results['metrics']['map']  # 예시로 mAP(metric) 사용
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params, best_score


param_dist = {
    'epochs': [20, 50, 100],
    'batch_size': [8, 16, 32],
    'learning_rate': [0.001, 0.01, 0.1],
    'img_size': [640, 1280]
}

best_model, best_params, best_score = random_search('data.yaml', param_dist, n_iter=10)
print(f'Best Params: {best_params}, Best Score: {best_score}')
