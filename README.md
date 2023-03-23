# Region-Aware Portrait Retouching with Sparse Interactive Guidance
[Huimin Zeng](https://hkchengrex.github.io/), Jie Huang, Jiacheng Li, Zhiwei Xiong

IEEE Transactions on Multimedia

[[arXiv]]( ) [[PDF]]( ) [[Project Page]]( ) [[Papers with Code]]( )
 
# Code comming soon
 
## Overview
<img src="fig/overview.png" height="260px"/> 
 

## Prerequisites
- Python 3.7
- Pytorch 1.8.1

To get started, first please clone the repo
```
git clone https://github.com/ZeldaM1/interactive_portrat_retouching.git
```
Then, please run the following commands:
```
conda create -n IPT python=3.7
conda activate IPT
pip install -r requirements.txt
```
You can also use our docker by running the following commands:
```
docker pull registry.cn-hangzhou.aliyuncs.com/zenghuimin/zhm_docker:py37-torch18
```

## Quick start
You can try our Demo!

1. Download the [[pre-trained models]( )]. 
3. Put the downloaded zip files to the root directory of this project
4. Run `bash prepare_data.sh` to unzip the files
5. Run the interactive portrait retouching demo
```bash
cd tool
python demo.py 
```
If everythings works, you will find an interactive GUI like:

<img src="materials/GUI.png" height="260px"/> 

You can also retouch your own portrait. All you need to do is to change the input and output paths, have fun!

## Todo list
- [ ] Update interactive demo
 
## Training
First, please prepare the dataset for training.
1. Please download [PPR10K dataset](https://github.com/csjliang/PPR10K) in the official link.
2. Unzip the PPR10K dataset to `./dataset`

Our codes follow a three-stage training process. In the first stage, we train the automatic branch.
```bash
cd interactive_portrat_retouching/codes
python train.py -opt options/train/stage1_automatic.yml
```
Then train the interactive branch.
```bash
cd interactive_portrat_retouching/codes
python train.py -opt options/train/stage2_interactive.yml
```

Third, train the joint model. 
```bash
cd interactive_portrat_retouching/codes
python train.py -opt options/train/stage3_joint.yml
```
 
## Citation
If our work inspires your research or some part of the codes are useful for your work, please cite our paper:
```bibtex

```

```bibtex

```

## Contact
If you have any questions, please contact us via 
- zenghuimin@mail.ustc.edu.cn

## Acknowledgement
Some parts of this repo are based on [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [CSRNet](https://github.com/hejingwenhejingwen/CSRNet).  
