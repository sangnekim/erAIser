# tobigs-image-conference
Video object segmentation branch  

base line code: [Siammask](https://github.com/foolwood/SiamMask)

## Contents
1. [Environment Setup](#environment-setup)
2. [Demo](#demo)
3. [Testing Models](#testing-models)
4. [Training Models](#training-models)

## Environment setup
This code has been tested on Ubuntu 18.04.5, Python 3.7, Pytorch 1.8.1, CUDA 10.2

- Clone the repository
pull 할거니까 이 부분 필요 없음
아래 shell 명령어들의 directory는 Siammask로 가정
- Setup python environment
```shell
conda create -n siammask python=3.7
conda activate siammask
pip install -r requirements.txt
```

## Demo
- [Setup](#environment-setup) your environment
- Download the SiamMask model
```shell
cd $SiamMask
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Run `demo.py`

```shell
python demo.py --resume SiamMask_DAVIS.pth --config config_davis.json
```

<div align="center">
  <img src="http://www.robots.ox.ac.uk/~qwang/SiamMask/img/SiamMask_demo.gif" width="500px" />
</div>

## Testing
- [Setup](#environment-setup) your environment
- Download test data (우리의 경우 Davis datset)
```shell
cd $SiamMask/data
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
ln -s ./DAVIS ./DAVIS2016
ln -s ./DAVIS ./DAVIS2017
```
- Download pretrained models
```shell
cd $SiamMask
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
- Evaluate performance on [DAVIS](https://davischallenge.org/) (less than 50s)
logging을 위한 logs 폴더 만들기
```shell
CUDA_VISIBLE_DEVICES=0 python -u test.py --config config_davis.json --resume SiamMask_DAVIS.pth --mask --refine --dataset DAVIS2017 2>&1 | tee logs/test_$dataset.log
```

<br>
Note:
- 기존의 siammask에 있는 많은 평가들은 vot에 해당하는게 많아서 여기서는 제외
- GPU 환경에서 돌릴 것!

## Training

### Training Data
- data directory의 각각 dataset 폴더에 들어가 [readme](data/coco/readme.md)에 나온대로 하면 데이터 설정 끝!
- 우리가 사용하는 데이터셋은 Youtube-VOS & COCO

### Download the pre-trained model (174 MB)
이 부분은 안해도 될 것 같음
이유 : resnet.py에서 따로 pytorch pre-trained를 불러옴
(This model was trained on the ImageNet-1k Dataset)
```shell
cd $SiamMask/experiments
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
ls | grep siam | xargs -I {} cp resnet.model {}
```

### Training SiamMask model with the Refine module
- [Setup](#environment-setup) your environment


- 경우1. 베이스 모델을 학습했거나 기존에 학습을 진행해서 checkpoint가 존재하는 경우 checkpoint 파라미터 사용
```shell
cd $SiamMask
bash python -u train.py --config=config.json -b 64 -j 20 --pretrained checkpoint_e12.pth --epochs 20 2>&1 | tee logs/train.log
```
- 경우2. checkpoint가 존재하지 않는 경우 siammask 저자들이 학습시켜 둔 데모에 사용하는 파라미터 사용
```shell
cd $SiamMask
bash python -u train.py --config=config.json -b 64 -j 20 --pretrained SiamMask_DAVIS.pth --epochs 20 2>&1 | tee logs/train.log
```
- You can view progress on Tensorboard (logs are at <experiment\_dir>/logs/)