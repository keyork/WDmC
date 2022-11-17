# WDmC: W(afer)D(efect)m(odes)C(lassifier)

Machine Learning & Big Data (2022 Fall) Final Project: Wafer Defect modes Classifier

**Assert!!!** To keep data confidential, we _**do not**_ upload the dataset file, please go to the [THU Web Learning](https://learn.tsinghua.edu.cn) to download the dataset by yourself.

## Envs

```
python==3.7
torch==1.8.0
torchvision==0.9.0
numpy==1.21.6
matplotlib==3.5.3
tensorboard==2.11.0
tqdm==4.64.1
colorlog==6.7.0
```

Create a python3.7 environment and run:

```
pip install -r requirements.txt
```

to install these packages in your environment.

---

_If you have some problems when installing `PyTorch`, these informations may help you._

To install `PyTorch(CPU)`, using: `pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`.

To install `PyTorch(GPU)`, you need to know the CUDA Version by using `nvcc -V`, we recommand `CUDA10.1` and `CUDA11.1`, other version may failure when installing, you can install CPU Version instead:

- `CUDA10.1`: `pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
- `CUDA11.1`: `pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`

## Usage

1. Make sure all path are exist, if not, create it:

```
cd data && mkdir processed && mkdir raw && cd ..
mkdir weights
```

2. Download `datasets2022.npz` and put it in `./data/raw/`

3. Train a model:

```
python run.py [-h] [--target TARGET] [--stage STAGE] [--rawpath RAWPATH]
              [--newpath NEWPATH] [--datadir DATADIR] [--trainscl TRAINSCL]
              [--bts BTS] [--lr LR] [--initmodel INITMODEL] [--loadwt LOADWT]
              [--weights WEIGHTS] [--saveweights SAVEWEIGHTS]
              [--dataset DATASET] [--epoch EPOCH] [--optim OPTIM]

optional arguments:
  -h, --help            show this help message and exit
  --target TARGET       train or eval
  --stage STAGE         project stage
  --rawpath RAWPATH     raw data path
  --newpath NEWPATH     new data path
  --datadir DATADIR     processed data path
  --trainscl TRAINSCL   train dataset scale
  --bts BTS             batch size
  --lr LR               learning rate
  --initmodel INITMODEL
                        init the model weights
  --loadwt LOADWT       load model weights
  --weights WEIGHTS     load model weights path
  --saveweights SAVEWEIGHTS
                        save model weights path
  --dataset DATASET     using dataset
  --epoch EPOCH         epoch num
  --optim OPTIM         optimizer(adam or sgd)
```

or run the script directly:

```
bash run.sh
```

4. Eval the model:

```
python run.py --target eval --initmodel False --loadwt True --weights ./weights/model13.pth
```

We provide a trained model weights here: https://github.com/keyork/WDmC/releases/tag/v1.0.0 or
https://cloud.tsinghua.edu.cn/f/7e40d23a599f49bd930e/

Download and put it in `./weights/`, then run this eval project.

## Eval

- `ACC`: `93.2435%`
- `AverageHammingDistance`: `0.010603`

## Our Model

TODO.
