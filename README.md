# erAIser - Remove an object in video using AI
<p>ppt 첫 슬라이드 이미지 넣기</p>

## Contents
1. [erAIser](#erAIser)
2. [Example](#Example)
3. [Demo screenshot](#Demo screenshot)
4. [Usage](#Usage)
    - Environment setup
    - Run demo
5. [References](#References)
6. [Contributors](#Contributors)

## erAIser 
<br>
<p>프로젝트 소개</p>
<br>

## Example
<br>
<p>원본 이미지 & 인페인팅 된 동영상 시연</p>
<br>

## Demo screenshot
<br>
웹 페이지 스크린샷
<br>

## Usage
### Caution

- This `root` directory is for 'integrated demo(Siammask + VINet)'.
- If you want to use 'video object segmentation(Siammask)' model only, you should go `vos` directory and follow `readme.md`
- If you want to use 'video inpainting(VINet)' model only, you should go `vi` directory and follow `readme.md`

### Environment Setup
This code was tested in the following environments
 - Ubuntu 18.04.5
 - Python 3.7
 - Pytorch 1.8.1
 - CUDA 9.0
 - GCC 5.5 (gnu c++14 compiler)

If you don't use gnu c++14 compiler, then you will encounter CUDA build error  

1. Clone the repository & Setup

```bash
git clone https://github.com/shkim960520/tobigs-image-conference.git
cd tobigs-image-conference
conda create -n erAIser python=3.7 -y
conda activate erAIser
conda install cudatoolkit=9.0 -c pytorch -y
pip install -r requirements.txt
bash install.sh
```

2. Setup python path

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
cd vos/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../vi/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../web/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../
```

### Demo

1. Setup your environment
2. Download the Deep Video Inpainting model

```bash
cd vi/results/vinet_agg_rec

file_id="1_v0MBUjWFLD28oMfsG6YwG_UFKmgZ8_7"
file_name="save_agg_rec_512.pth"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

file_id="12TRCSwixAo9AqyZ0zXufNKhxuHJWG-RV"
file_name="save_agg_rec.pth"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}

cd ../../../
```
3. Download the Siammask model

```bash
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```

4. Make `results` directory for saving result video
```bash
mkdir results
```
`results` is defualt setting. You can change this.

5. Run `inference.py`
```bash
python3 inference.py --resume SiamMask_DAVIS.pth --config config_inference.json
```
The result video will be saved in `results`.

## References
<p>Siammask 논문 등등</p>

## Contributors
<p>웹에 들어가는 members 그대로 사용하기</p>
