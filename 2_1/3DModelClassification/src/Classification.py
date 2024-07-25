import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image

# 클래스 정의
classes = ['GLTD', 'IRP', 'LRF']

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.base_model = models.vgg16(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_features = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

model = MyModel(num_classes=len(classes))

# 모델 로드
state_dict = torch.load('model10epoch_64batch.pth')
model.load_state_dict(state_dict)
model.eval()

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 이미지 로드
image = Image.open('TEST_LRF.jpg')
image = transform(image).unsqueeze(0)

# 추론 수행
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    print('Predicted class:', classes[predicted.item()])