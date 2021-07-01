# erAIser
## tobigs-image-conference


## Caution

- This `root` directory is for 'integrated demo(siammask + VINet)'.
- If you want to know about 'video object segmentation(siammask)' model only, you should go `vos` directory
- If you want to know about 'videon inpainting(VINet)' model only, you should go `vi` directory

## Contents

1. Environment Setup
2. Demo

## Environment Setup
This code has been tested on Ubuntu 18.04.5, Python 3.7, Pytorch 1.8.1 CUDA 9.0, c++14  
If you don't use gnu c++14 compiler, then you will encounter CUDA build error  

- Clone the repository
- Setup python environment

```bash
git clone https://github.com/shkim960520/tobigs-image-conference.git
cd tobigs-image-conference
conda create -n erAIser python=3.7 -y
conda activate erAIser
conda install cudatoolkit=9.0 -c pytorch -y
pip install -r requirements.txt
bash install.sh
```

- Setup python path

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
cd vos/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../vi/
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../
```

## Demo

- Setup your environment
- Download the Deep Video Inpainting model

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
- Download the Siammask model

```bash
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```

- Run `inference.py`

```bash
python3 inference.py
```

- You can see the result under results folder
