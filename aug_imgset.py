import imgaug.augmenters as iaa
import glob
import cv2
import os
import numpy as np

def augment_images(images):
    # 모든 이미지를 320x320 크기로 변경
    resize = iaa.Resize((320, 320))

    # 증강 정의
    augmenter_seq = iaa.Sequential([
        # 수평 뒤집기 적용
        iaa.Fliplr(1.0), 
        # 수직 뒤집기 적용
        iaa.Flipud(1.0),
        # 밝기 조절 (-30% ~ +30%)
        iaa.Multiply((0.7, 1.3)),
        # 노출 조절 (-10% ~ +10%)
        iaa.LinearContrast((0.9, 1.1)),
    ])

    augmented_images = []
    for image in images:
        image_resized = resize(image=image)
        for angle in [0, 90, 180]:
            rotate = iaa.Rotate(angle)
            final_augmenter = iaa.Sequential([rotate, augmenter_seq])

            augmented_images.append(final_augmenter(image=image_resized))

    return np.array(augmented_images)

input_path = ''     # 원본 이미지 디렉토리 경로
output_path = ''    # 출력 디렉토리 경로

""""
ex )
input_path = "C:\\Users\\Android\\Desktop\\common_road\\*.*"
output_path = "C:\\Users\\Android\\Desktop\\aug_common_road\\"
"""

# 출력 경로가 없으면 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 입력 경로에서 모든 이미지 파일 가져오기
image_files = glob.glob(input_path)

for idx, image_file in enumerate(image_files):
    image = cv2.imread(image_file)
    augmented_images = augment_images([image])

    for aug_idx, aug_image in enumerate(augmented_images):
        """
        output_filename = os.path.join(output_path, f"common_road_{idx}_{aug_idx}.jpg")
        output_filename = os.path.join(output_path, f"mud_flooded_road_{idx}_{aug_idx}.jpg")
        output_filename = os.path.join(output_path, f"nonmud_flooded_road_{idx}_{aug_idx}.jpg")
        """
        cv2.imwrite(output_filename, aug_image)
