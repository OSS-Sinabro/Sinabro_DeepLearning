<div align="center">
  <h1>Sinabro DeepLearning</h1>

  <table>
      <tr>
          <td align="center">침수 분류모델</td>
          <td align="center"><a href="https://drive.google.com/file/d/16JeA2ZvXkhJcd5dfkVBkT9tbOrz0xvyb/view?usp=sharing">Flooded_road_classification_model.pth</a></td>
      </tr>
      <tr>
          <td align="center">학습 데이터셋</td>
          <td align="center"><a href="https://drive.google.com/file/d/1tS9qnNAwa5reUW6AdTwh51Xe_phw_6dZ/view?usp=sharing">dataset.zip</a></td>
      </tr>
  </table>
</div>

<br/>

## ⚙ 학습 환경
>![학습환경](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/assets/90829718/f7da4fd3-04a5-44c9-aefe-784c20158533)

<br/>

## 💡 학습 결과

| Loss Graph | Precision / Recall / F1 score |
| :--------: | :---------------------------: |
| ![손실그래프 이미지](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/assets/90829718/761c903e-c91f-4dd4-8539-09590c77f624) | ![정밀도, 재현율, F1 스코어 이미지](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/assets/90829718/eb4c8877-edc1-45b2-9676-d4359a1cda8c) |

| <div align="center" colspan="2">Test-set 검증</div> |
| :--------: |
| ![테스트 이미지](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/assets/90829718/7ca6e1b4-bd82-400e-9963-d60850cdc9bb) |

<br/>

## 📚 데이터셋 구축 프로세스

| **데이터 확보** | **데이터 증강** |
|:--------:|:--------:|
| ![데이터 확보](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/assets/90829718/445957a7-54d7-49a6-a223-e7546dd53abf) | ![데이터 증강](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/assets/90829718/f443e1fc-0360-479c-86a3-65867e9fabfc) |
|   | [aug_imgset.py](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/blob/main/aug_imgset.py) |
| **데이터 분할** | **csv 생성** |
| ![분할](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/assets/90829718/93342ecb-c74a-4766-a977-0ffacffa5885) | ![csv (1)](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/assets/90829718/a2a1874e-8efe-4018-b01a-014ed927f870) |
| [divide_imgset.py](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/blob/main/divide_imgset.py) | [make_csv.py](https://github.com/OSS-Sinabro/Sinabro_DeepLearning/blob/main/make_csv.py) |




