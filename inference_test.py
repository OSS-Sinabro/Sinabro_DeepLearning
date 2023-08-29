from torchvision.models import resnet152
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
import os
import time
import torch.nn.functional as F

# 모델 정의
model = resnet152()
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 3)
)

# 저장된 모델 가중치 로드
model_path = 'Flooded_road_classification_model.pth' # 모델 경로 입력
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


# 이미지 추론 함수
def infer_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_probability, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), predicted_probability.item() * 100  # 추론 신뢰도


# 이미지 파일 확장자 목록
image_extensions = ['.jpg', '.jpeg', '.png']

# 입력 디렉토리 경로
input_directory_path = '' # 추론할 이미지가 위치한 디렉토리 경로 입력

# 입력 디렉토리 내의 이미지 파일만 읽어옴
image_files = [f for f in os.listdir(input_directory_path) if os.path.isfile(os.path.join(input_directory_path, f)) and os.path.splitext(f)[1].lower() in image_extensions]

# 각 이미지 파일에 대해 추론 수행
for image_file in image_files:
    image_path = os.path.join(input_directory_path, image_file)
    start_time = time.time()
    predicted_class, predicted_probability = infer_image(image_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{image_file} : [ Class {predicted_class} ]  with {predicted_probability:.2f}% confidence (Inference time: {elapsed_time:.4f} seconds)")

