import os
import shutil
import random

def split_data(source_dir, training_dir, validation_dir, test_dir, ratio=(0.8, 0.1, 0.1)):
    files = os.listdir(source_dir)
    random.shuffle(files)

    training_count = int(len(files) * ratio[0])
    validation_count = int(len(files) * ratio[1])

    for i, file in enumerate(files):
        source = os.path.join(source_dir, file)
        if i < training_count:
            destination = os.path.join(training_dir, file)
        elif i < training_count + validation_count:
            destination = os.path.join(validation_dir, file)
        else:
            destination = os.path.join(test_dir, file)

        shutil.copyfile(source, destination)

# 이미지 데이터 분할 후, 저장할 디렉토리 경로
train_root = ''
val_root = ''
test_root = ''
""""
ex )
train_root = "C:\\Users\\Android\\Desktop\\Sinabro_dataset\\train"
val_root = "C:\\Users\\Android\\Desktop\\Sinabro_dataset\\val"
test_root = "C:\\Users\\Android\\Desktop\\Sinabro_dataset\\test"
"""


# 일반도로
common_root = ''    # 증강 완료한 이미지데이터 위치 디렉토리 경로
split_data(common_root, train_root, val_root, test_root)

# non황토색침수도로
nonmud_flooded_root = ''    # 증강 완료한 이미지데이터 위치 디렉토리 경로
split_data(nonmud_flooded_root, train_root, val_root, test_root)

# 황토색침수도로 이미지
mud_flooded_root = ''   # 증강 완료한 이미지데이터 위치 디렉토리 경로
split_data(mud_flooded_root, train_root, val_root, test_root)

"""
ex )
common_root = "C:\\Users\\Android\\Desktop\\aug_common_road"
nonmud_flooded_root = "C:\\Users\\Android\\Desktop\\aug_nonmud_flooded_road"
mud_flooded_root = "C:\\Users\\Android\\Desktop\\aug_mud_flooded_road"
"""

