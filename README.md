# VRDL HW1 Demo 

## Reference 

This project is modified from 
- Github repo: 
 [Code release for Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches (ECCV2020)](https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training)
- Original paper:  
 [Fine-Grained Visual Classification via Progressive Multi-Granularity Training of Jigsaw Patches
Ruoyi Du, Dongliang Chang, Ayan Kumar Bhunia, Jiyang Xie, Zhanyu Ma, Yi-Zhe Song, Jun Guo](https://arxiv.org/abs/2003.03836)


## Reproduce answer.txt

### Requirement
- python 3.6

- PyTorch >= 1.3.1

- torchvision >= 0.4.2
### Training Steps

1. Download datatsets for HW1 and organize the structure as follows:
```

└── VRDL_HW1_DEMO
    ├── datasets
    │   └── CUB
    │       ├── classes.txt
    │       ├── testing_images
    |       |    ├──xxxx.jpg
    |       |    ├──xxxx.jpg
    |       |    ...
    |       |    └──xxxx.jpg
    │       ├── testing_img_order.txt
    │       ├── training_images
    |       |    ├──xxxx.jpg
    |       |    ├──xxxx.jpg
    |       |    ...
    |       |    └──xxxx.jpg
    │       └── training_labels.txt
    ...
    (other files)
```

2. Train from scratch with ``train.py``.
(training takes about 5~6 hours on single gpu)


```bash
cd VRDL_HW1_DEMO
python train.py
```

3. To reproduce my submition without training:  
   Download my trained `model.pth` from [google_drive](https://drive.google.com/file/d/1yniVOaTM_FUp6WCeReMALP4o5vmAXsGJ/view?usp=sharing)


3. After training, model parameters will be stored inside a directory name `experiment_time=xxxxxxxxx`

```
.
└── VRDL_HW1_DEMO
    ├── expiriment_time=Thu-Nov--4-19:24:03-2021
    │   ├── model.pth
    │   ├── results_test.txt
    │   └── results_train.txt
     ...
    (other files)
```

5. run `inference.py` and provide the `model.pth` file to generate `answer.txt`
```bash
python inference.py --model_path=expiriment_time=Thu-Nov--4-19:24:03-2021/model.pth
```
6. The desired `answer.txt` will be generated under the same folder as `model.pth`
(the filename is `eval_result11-04-2021__20:47:15.txt` in this case) 

```
.
└── VRDL_HW1_DEMO
    ├── expiriment_time=Thu-Nov--4-19:24:03-2021
    │   ├── model.pth
    │   ├── eval_result11-04-2021__20:47:15.txt
    │   ├── results_test.txt
    │   └── results_train.txt
     ...
    (other files)
```

