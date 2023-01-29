# BoxDetection-DINO

# 사용방법
## 설치방법 1
```
https://github.com/KAI-Devv/BoxDetection-DINO.git<br>
cd BoxDetection-DINO <br>
```
https://pytorch.org/get-started/locally/ 에서 개발 환경에 맞는 pytorch와 torchvision을 설치합니다.
설치가 완료되면 나머지 라이브러리르 아래 스크립트를 통해 설치합니다.
```
pip install -r requirements.txt <br>
```
CUDA Compile을 위해 아래 스크립트를 수행합니다.
```
cd models/dino/ops
python setup.py build install
```
아래 스크립트를 통해 Compile이 되었는지 확인합니다.
```
python.test.py
```
```
cd ../../..
```

## 설치방법 2 (Docker를 활용한 설치)
아래 링크를 다운로드 하여 도커 설치를 따라해주시면 됩니다.
1) 철도 선로 데이터셋 : https://github.com/KAI-Devv/BoxDetection-DINO/files/10401838/_DINO_225.docx
2) 전차선 / 애자 데이터셋 : https://github.com/KAI-Devv/BoxDetection-DINO/files/10401839/_DINO_226.docx <br>


## 데이터 전처리
데이터 전처리 자동화 스크립트를 통해 DINO에서 학습 가능한 라벨렝 데이터 포맷으로 변환하고,
Training:Validation:Test 데이터 비율을 8:1:1로 랜덤하게 분배하여 저장합니다.
탐지 객체 이름과 id 정보가 포함된 데이터셋 별 메타데이터 파일은 해당 파일을 참고 바랍니다.
1) 철도 선로 데이터셋 : preprocess/railway_metadata.json<br>
2) 전차선 / 애자 데이터셋 : preprocess/catenary_metadata.json><br><br>

아래 스크립트를 통해 전처리를 수행합니다.<br>
```
python preprocess/preprocess.py {데이터셋 폴더 경로} {메타데이터 파일 경로}<br><br>
```
해당 스크립트가 수행되면 {데이터셋 폴더 이름}에 '_data'라는 suffix를 가진 폴더가 생성되고. Training:Validation:Test 데이터를 분리합니다. <br>

아래 스크립트를 통해 두번째 전처리를 수행합니다.<br>
```
python preprocess/preprocess2.py {데이터셋 폴더 경로}_data {메타데이터 파일 경로}<br><br>
```
이는 yolov5 포맷에 맞는 라벨링 데이터 .txt 파일을 이미지별로 생성하고, 클래스 정보를 포함한 내용을 {데이터셋 폴더 이름}_data/data.yaml 파일에 저장됩니다.<br>

예시)
1) 철도 선로 데이터셋 <br>
    ```
    python preprocess/preprocess.py ../../../dataset/railway preprocess/railway_metadata.json <br>
    python preprocess/preprocess2.py ../../../dataset/railway_data preprocess/railway_metadata.json
    ```
2) 전차선 / 애자 데이터셋 <br>
    ```
    python preprocess/preprocess.py ../../../dataset/catenary preprocess/catenary_metadata.json <br>
    python preprocess/preprocess2.py ../../../dataset/catenary_data preprocess/catenary_metadata.json
    ```

## 학습
학습을 위해 아래 스크립트 포맷 및 예시를 참고합니다.

```
python -m torch.distributed.launch --nproc_per_node={GPU 개수} main.py \
	--output_dir logs/{로그 저장 경로} -c config/DINO/DINO_4scale.py --coco_path {preprocess2.py를 통해 획득된 학습 데이터셋 경로} \
	--pretrain_model_path {.pth 확장자의 모델 경로} \
	--finetune_ignore label_enc.weight class_embed \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 num_classes={클래스 수} dn_labelbook_size={클래스 수}
```
예시)
1) 철도 선로 데이터셋 <br>
```
python -m torch.distributed.launch --nproc_per_node=1 main.py \
	--output_dir logs/exp_railway -c config/DINO/DINO_4scale.py --coco_path ../../../dataset/railway_data \
	--pretrain_model_path railway.pth \
	--finetune_ignore label_enc.weight class_embed \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 num_classes=58 dn_labelbook_size=58
```
2) 전차선 / 애자 데이터셋 <br>
```
python -m torch.distributed.launch --nproc_per_node=1 main.py \
	--output_dir logs/exp_catenary -c config/DINO/DINO_4scale.py --coco_path ../../../dataset/catenary_data \
	--pretrain_model_path catenary.pth \
	--finetune_ignore label_enc.weight class_embed \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 num_classes=40 dn_labelbook_size=40
```

## 유효성 검증
유효성 검증은 아래 스크립트 포맷 및 예시를 참고합니다.
```
python validate-railway-dataset.py --output logs/{로그 저장 경로} -c config/DINO/DINO_4scale.py --coco_path {preprocess2.py를 통해 획득된 학습 데이터셋 경로} --eval --resume railway_dio.pth --data_type {railway or catenary} --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 batch_size 1 num_classes={클래스 수} dn_labelbook_size={클래스 수}
```
예시)
1) 철도 선로 데이터셋 <br>
```
python validate-railway-dataset.py --output logs/exp_railway -c config/DINO/DINO_4scale.py --coco_path ../../../dataset/railway_data --eval --resume railway_dino.pth --data_type railway --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 batch_size 1 num_classes=58 dn_labelbook_size=58
```
2) 전차선 / 애자 데이터셋 <br>
```
python validate-railway-dataset.py --output logs/exp_catenary -c config/DINO/DINO_4scale.py --coco_path ../../../dataset/catenary_data --eval --resume catenary_dino.pth --data_type catenary --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 batch_size 1 num_classes=40 dn_labelbook_size=40
```

<br><br><br>

# 모델 정보
## 모델 Description
DINO 모델 <br>

## 모델 아키텍쳐
<img width="850" src="https://github.com/abbasmammadov/DINO/raw/main/figs/framework.png"></a>

## 모델 입력값
모델 입력값으로 (B, H, W, 3)의 Tensor를 사용합니다. B는 Batch Size, H는 높이, W는 넓이값입니다. <br>

## 모델 출력값
모델 출력값으로 (6, 1) Shape의 바운딩박스, Confidence, Class 번호가 리스트형태로 출력됩니다. <br>
각 텐서값은 순서대로 (min_x, min_y, max_x, max_y, confidence, class_num)를 정의합니다.

## 모델 TASK
객체 탐지 <br>

## Training Dataset
학습 데이터는 이미지 파일과 이미지 파일에서 감지할 객체의 라벨링 데이터입니다.
이미지 파일은 .jpg 또는 .png 파일을 사용합니다. 라벨링 데이터는 .txt 파일 형태이며, 카테고리 id와 normalized된 center x, y 및 width, height값으로 구성됩니다.
preprocess.py 및 preprocess2.py 스크립트를 이용하여 철도 선로 및 전차선/애자 데이터 셋을 해당 포맷으로 자동으로 변환시켜 주어야 합니다.

## Configurations
학습을 위해 batch size는 64, SGD optimizer, BCE + CIoU Loss, learning rate 0.01을 사용하였습니다.

## Evaluation
mAP (0.75 IoU 기준)
1) 철도 선로 데이터셋 : 0.8394
2) 전차선 / 애자 데이터셋 : 0.8204

## Reference
YOLOv5모델을 사용하였으며, 학습데이터 적용 및 유효성 검증 수행을 위한 코드의 커스터마이징 작업이 수행되었습니다. <br>
관련 링크: https://github.com/IDEA-Research/DINO

## License
SPDX-FileCopyrightText: © 2022 Seunghwa Jeong <<sh.jeong@kaistudio.co.kr>> <br>
SPDX-License-Identifier: Apache-2.0 <br>

YOLOv5에서 명시된 Apache 2.0 License를 따릅니다. <br>
(Apache 2.0 전문: https://www.apache.org/licenses/LICENSE-2.0)