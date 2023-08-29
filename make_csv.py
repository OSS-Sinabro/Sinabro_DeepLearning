import os
import pandas as pd

## 이미지 파일들이 위치한 디렉토리 경로 설정
"""
# train
directory_path = ''
# ex ) directory_path = 'C:\\Users\\Android\\Desktop\\Sinabro_dataset\\train'

# val
directory_path = ''
# ex ) directory_path = 'C:\\Users\\Android\\Desktop\\Sinabro_dataset\\val'

# test
directory_path = ''
# ex ) directory_path = 'C:\\Users\\Android\\Desktop\\Sinabro_dataset\\test'
"""

# 이미지 파일 확장자
image_extensions = ['.jpg', '.jpeg', '.png']

# CSV 파일 생성
df = pd.DataFrame(columns=['filename', 'common_road', 'mud_flooded_road', 'nonmud_flooded_road'])

# 디렉토리 내의 모든 파일 검색
for filename in os.listdir(directory_path):
    # 이미지 파일만 선택
    if any(filename.endswith(ext) for ext in image_extensions):
        # 파일명 'common_road' 포함 -> 일반도로
        common_road = 1 if 'common_road' in filename else 0
        # 파일명 'mud_flooded_road' 포함 & 'non' 미포함 -> 황토 침수도로
        mud_flooded_road = 1 if 'mud_flooded_road' in filename and 'non' not in filename else 0
        # 파일명 'mud_flooded_road' 포함 -> 비황토 침수도로
        nonmud_flooded_road = 1 if 'non' in filename else 0

        # 파일명으로 구별하여 해당 이미지에 대해 common, mud, nonmud 중 1클래스 부여, 나머지는 0 부여
        df.loc[len(df)] = [filename, common_road, mud_flooded_road, nonmud_flooded_road]


## CSV 파일 저장할 디렉토리 경로 설정
"""
# train
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(f'{directory_path}\\train.csv', index=False)

# val
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(f'{directory_path}\\val.csv', index=False)

# test
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(f'{directory_path}\\test.csv', index=False)
"""