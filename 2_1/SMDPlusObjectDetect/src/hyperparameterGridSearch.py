from sklearn.model_selection import ParameterGrid
from ultralytics import YOLO


def grid_search(data, param_grid):
    best_model = None
    best_score = float('-inf')
    best_params = None

    for params in ParameterGrid(param_grid):
        model = YOLO('yolov8n.pt')

        # YOLOv8 학습 인자 형식 맞추기
        yolo_params = {
            'data': data,
            'epochs': params['epochs'],
            'batch': params['batch_size'],
            'lr0': params['learning_rate']
        }

        results = model.train(**yolo_params)

        # 예시로 mAP 사용 (적절한 평가 지표로 교체 가능)
        score = results.metrics.mAP50
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params

    return best_model, best_params, best_score


param_grid = {
    'epochs': [2, 10],
    'batch_size': [8, 64],
    'learning_rate': [0.001, 0.01],
}
if __name__ == '__main__':
    best_model, best_params, best_score = grid_search('data.yaml', param_grid)
    print(f'Best Params: {best_params}, Best Score: {best_score}')
