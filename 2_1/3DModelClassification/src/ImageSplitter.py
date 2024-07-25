import os
import numpy as np
import shutil

# 원본 데이터셋 경로
image_folder = "../datasets/texture_datasets"

# 증강된 데이터셋 경로
output_folder = "../datasets/final_texture_datasets"
subfolders = ["train", "val", "test"]
classes = ["LRF", "GLTD", "IRP"]

ratios = [0.65, 0.35, 0]

def split_image():
    for cls in classes:
        real_image_path = os.path.join(image_folder, cls)
        files = os.listdir(real_image_path)
        
        np.random.shuffle(files)
        
        # 각 분할에 대한 파일 수를 계산합니다.
        split_sizes = [int(len(files)*ratio) for ratio in ratios]

        # 분할 크기의 합이 파일의 총 수와 같은지 확인합니다.
        split_sizes[-1] = len(files) - sum(split_sizes[:-1])

        # 파일을 분할합니다.
        split_files = np.split(files, np.cumsum(split_sizes)[:-1])

        # 분할된 파일을 대상 디렉토리에 저장합니다.
        for folder, files in zip(subfolders, split_files):
            dest_path = os.path.join(output_folder, folder, cls)
            os.makedirs(dest_path, exist_ok=True)
            for file in files:
                shutil.copy(os.path.join(real_image_path, file), dest_path)
                

split_image()