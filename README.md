# WDMC

W(afer)D(efect)M(odes)C(lassifier)

Machine Learning & Big Data (2022 Fall) Final Project: Wafer Defect Modes Classifier

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

To install `PyTorch(GPU)`, you need to know the CUDA Version by using `nvcc -V`, we recommand `CUDA10.1` and `CUDA11.1`, other version may failure when installing you can install CPU Version instead:

- `CUDA10.1`: `pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
- `CUDA11.1`: `pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
