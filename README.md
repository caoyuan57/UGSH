# UGSH
---

## Requirements

* Python == 3.8.13
* PyTorch == 2.1.0
* Torchvision == 0.16.0
* Transformers == 4.4.2

## Configs

You can find the configurations in `./configs/base_config.py`:

* CUDA device
* seed
* data and dataset paths
---


## How to run

```
main.py [-h] [--test] [--bit BIT] [--model MODEL] [--epochs EPOCHS]
        [--dataset DATASET] [--preset PRESET]

optional arguments:
  -h, --help            show this help message and exit
  --test                run test
  --bit BIT             hash code length
  --model MODEL         model type
  --epochs EPOCHS       training epochs
  --dataset DATASET     ucm or rsicd
  --preset PRESET       data presets, see available in config.py
```

train and test:

1. Train a model for 64-bit hash codes generation using the RSICD dataset and the default data preset:
```
python main.py --dataset rsicd --preset default --bit 64
```

2. Run a test for the model from previous example:
```
python main.py --dataset rsicd --preset default --bit 64 --test
```

---

