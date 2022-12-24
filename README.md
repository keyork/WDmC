# WDmC: W(afer)D(efect)m(odes)C(lassifier)

🎉🎉🎉 We have finally completed this course project!!!

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
mkdir data && cd data && mkdir processed && mkdir raw && mkdir result && cd ..
mkdir weights && mkdir runs
```

**Update**: the program can create the dir automatically.

2. Download `datasets2022.npz` and put it in `./data/raw/`

3. Train a model:

```
usage: run.py [-h] [--target TARGET] [--stage STAGE] [--rawpath RAWPATH]
              [--newpath NEWPATH] [--datadir DATADIR] [--trainscl TRAINSCL]
              [--bts BTS] [--lr LR] [--initmodel INITMODEL] [--loadwt LOADWT]
              [--weightsroot WEIGHTSROOT] [--weights WEIGHTS] [--model MODEL]
              [--saveweights SAVEWEIGHTS] [--dataset DATASET] [--epoch EPOCH]
              [--optim OPTIM] [--result RESULT] [--final FINAL]

optional arguments:
  -h, --help            show this help message and exit
  --target TARGET       train or eval
  --stage STAGE         project stage, self/raw-train/final-test
  --rawpath RAWPATH     raw dataset path
  --newpath NEWPATH     new dataset path
  --datadir DATADIR     processed dataset path
  --trainscl TRAINSCL   train dataset scale
  --bts BTS             batch size
  --lr LR               learning rate
  --initmodel INITMODEL
                        init the model weights (True/False)
  --loadwt LOADWT       load model weights (True/False)
  --weightsroot WEIGHTSROOT
                        load model weights root path
  --weights WEIGHTS     load model weights path
  --model MODEL         model type (xvgg16/xresnet50/xvit)
  --saveweights SAVEWEIGHTS
                        save model weights path
  --dataset DATASET     using dataset
  --epoch EPOCH         epoch num
  --optim OPTIM         optimizer (adam or sgd)
  --result RESULT       result file path
  --final FINAL         is the last one (True/False)
```

or run the script directly:

```
bash run.sh
```

4. Eval the model:

```
python -u run.py --target eval --initmodel False --loadwt True --weightsroot ./weights --weights model-full-final-2.pth --model neck
```

or get the result file (.csv):

```
python -u run.py --target eval --stage final-test --initmodel False --loadwt True --weightsroot ./weights --weights model-full-final-2.pth --model neck --result ./data/result/Group5.csv
```

We provide a trained model weights here: https://github.com/keyork/WDmC/releases/tag/v2.0.0

Download and put it in `./weights/`, then run this eval project.

## Eval

- `ACC`: `98.6191%`
- `AverageHammingDistance`: `0.002199`

## Our Model

We train three different models independently, and use a neck model to combine them.

![model-img](./data/readme-img/model-img.png)

### Transform

- Transform 1
  - train
    - RandomHorizontalFlip, p=0.4
    - RandomVerticalFlip, p=0.4
    - RandomRotation, degree=45
    - Normalize, mean=0.5, std=0.5
  - test
    - Normalize, mean=0.5, std=0.5
- Transform 2
  - train
    - Resize -> (224,224)
    - RandomHorizontalFlip, p=0.4
    - RandomVerticalFlip, p=0.4
    - RandomRotation, degree=45
    - Normalize, mean=0.5, std=0.5
  - test
    - Resize -> (224,224)
    - Normalize, mean=0.5, std=0.5

Above all, Transform 2 is the same as Transform 1 except **resize**

### Model ABC

Model A is based on VGG16, Model B is based on ResNet50, Model C is based on Vit(Transformer).

We cut the model from the last two layer and get a (1x64) feature for each, and these three features can be used in the next stage.

### Neck Model

![neck-img](./data/readme-img/neck-img.png)

## Train Method

### Fake Test

The raw dataset doesn't contain the label of test data, but we want to evaluate our model just like the final test, so we make fake test dataset from raw dataset.

To make sure the model can't get in touch with test data while training, we random select 20\% data from raw dataset as the fake test data, others are still train dataset.

### Split Dataset

We found that it is really hard to train the model on the full dataset directly, after exploring the dataset, we found the defects in one img is not the same, this means IMG-A may contains 1 defect, but IMG-B may contains 6 defects. This can be a big deal for deep learning model.

So we select the image which only contains 0 or 1 defect as base-dataset, which contains 2 defects as double-dataset, others as multi-dataset. And then, we get bd-dataset (base & double) by add base-dataset and double-dataset.

While training, we follow the order:

1. base-dataset
2. double-dataset
3. bd-dataset
4. multi-dataset
5. full-dataset

remark: full-dataset is the dataset before spliting.

### Loss, Optim and LR Scheduler

We use MSELoss from beginning to end, as for optimizer, we use Adam most of the time, and use SGD only in the last training epochs group.

We use lr_scheduler according to the hamming distance on valid dataset, when hamming distance no longer drops more than 4 epochs, we set the learning rate as 0.1\*old learning rate.

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.1,
        patience=4,
        verbose=False,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=3,
        min_lr=1e-7,
    )
```

### Whole Training Process

1. Train three model using `base -> double -> bd -> multi -> full` dataset on Adam optim
2. Continue training them on SGD optim
3. Load three trained model, and remove the last two layers from each of them
4. Train Neck-Model like 1-2

### Train Scalars

We use tensorboard to record the `[accuracy, hamming_distance, loss]` of train set and valid set while training, here are some images.

![acc-full](./data/readme-img/acc-full.png)
![dts-full](./data/readme-img/dts-full.png)
![loss-full](./data/readme-img/loss-full.png)

And here is one record in them, orange line is train-set and blue line is valid-set.

![acc-1](./data/readme-img/acc-1.png)
![dts-1](./data/readme-img/dts-1.png)
![loss-1](./data/readme-img/loss-1.png)
