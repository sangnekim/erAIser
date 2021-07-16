# erAIser - Remove an object in video using AI
<p align="center"><img width="539" alt="첫슬라이드" src="https://user-images.githubusercontent.com/40483474/125912276-4d5b8952-7973-4884-80ff-93f475fb3bb8.PNG">
</p>

## Contents
1. [erAIser](#erAIser)
2. [Example](#Example)
3. [Demo screenshot](Demo-screenshot)
4. [Usage](#Usage)
    - Environment setup
    - Run demo
5. [References](#References)
6. [Contributors](#Contributors)

## erAIser 
<br>
‘erAIser’ is a service that provides a specific object erased video by using video object segmentation and video inpainting methods.
<br>

Most of video inpainting model need segmentation mask of objects. But it is hard to get in normal way. For your convenience, we used a deep learning model that allows users to easily obtain segmentation masks. Also, we combined this video object segmentation model with the video inpainting models to increase usability.

Our team consists of nine members of ‘Tobigs’ who are interested in computer vision task.

Let’s make your own video of a specific object being erased with ‘erAIser’!

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
cd ../AANet/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../
```

### Demo

1. Setup your [environment](#Environment-setup)
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

file_id="1IKZWpMeWXq-9osBqG7e7bTABumIZ32gB"
file_name="checkpoint_e19.pth"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}
```

4. Make `results` directory for saving result video
```bash
mkdir results
```
`results` is defualt setting. You can change this.

5-1. Run `inference.py` for erasing
```bash
python3 inference.py --resume checkpoint_e19.pth --config config_inference.json
```

5-2. Run `inference.py` for change people
```bash
python3 inference.py --resume SiamMask_DAVIS.pth --config config_inference.json --using_aanet True
```
The result video will be saved in `results`.

## References
<p>Wang, Qiang, et al. "Fast online object tracking and segmentation: A unifying approach." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.</p>
<p>Wang, Tianyu, et al. "Instance shadow detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.</p>

## Contributors
<p>웹에 들어가는 members 그대로 사용하기</p>
