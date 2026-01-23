### Installation
```
Requirements:
PyTorch == 1.10.0
ignite == 0.2.1
torchvision == 0.11.2
```
### Prepare data
The file tree should be
```
dataset
└── LLCM
    └── idx
    └── nir
    └── vis
    └── test_nir
    └── test_vis
└── SYSU-MM01
    └── cam1
    └── cam2
    └── cam3
    └── cam4
    └── cam5
    └── cam6
    └── exp
```

### Prepare Source Pre-trained Models
```
shell
python -m main.train --cfg ${config} --gpu ${gpu_id} --model ${method}  --output ${log path}
# e.g., python -m main.train --cfg ./configs/SYSU.yml --gpu 0 --model resnet --output train
```


### Perform Tset-time-adaptation
```
shell
./inference/${dataset}.sh ${description} ${checkpoint}
# e.g., sh ./inference/market.sh test model_best
python -m main.adapt --cfg ${config} --gpu ${gpu_id} --resume ${checkpoint path} --model ${method}  --output ${log path}
python -m main.adapt --cfg ./configs/SYSU.yml --gpu 0 --resume ./checkpoints/ckpt.pth --model resnet --output adaptation
```
