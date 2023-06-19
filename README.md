<!-- HEADER -->
<p align="center">
    <h1 align="center">Object pop-up: Can we infer 3D objects and their poses from human interactions alone?</h1>
    <!-- authors -->
    <p align="center">
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/Petrov.html"><b>Ilya A. Petrov</b></a>
        &emsp;
        <a href="https://riccardomarin.github.io/"><b>Riccardo Marin</b></a>
        &emsp;
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/Chibane.html"><b>Julian Chibane</b></a>
        &emsp;
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html"><b>Gerard Pons-Moll</b></a>
    </p>
    <!-- conference -->
    <h3 align="center">CVPR 2023</h3>
    <!-- teaser -->
    <p align="center">
        <img src="assets/petrov23popup.gif" alt="Project Teaser" width="600px">
    </p>
    <!-- badges -->
    <p align="center">
        <a href="https://arxiv.org/abs/2306.00777">
            <img src="https://img.shields.io/badge/arXiv-2306.00777-b31b1b.svg?style=for-the-badge" alt="Paper PDF">
        </a>
        &emsp;
        <a href="https://virtualhumans.mpi-inf.mpg.de/object_popup/">
            <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=Google%20chrome&logoColor=white" alt="Project Page">
        </a>
        &emsp;
        <a href="https://youtu.be/qMDcSIXQgR0">
            <img src="https://img.shields.io/badge/YouTube-video-black?style=for-the-badge&logo=youtube&logoColor=white&labelColor=FF0000&color=black" alt="YouTube video">
        </a>
    </p>
</p>


## Environment
The code was tested under `Ubuntu 22.04, Python 3.10, CUDA 11.8, PyTorch 2.0.1`.\
Use the following command to create a conda environment with necessary dependencies:
```bash
conda env create -f environment.yml
```


## Data downloading and processing
The steps are described in [docs/data.md](./docs/data.md).


## Pre-trained models and evaluation
Pre-trained models can be obtained from the [link](). With the commands:
```bash
wget -O ./assets/object_pop_up_noclass.pth 
wget -O ./assets/object_pop_up_class.pth
```

Use the following commands to run evaluation:
```bash
# model without class prediction (assumes 24GB GPU memory)
python evaluate.py scenarios/gb_PNv2_noclass.toml -b 64 -w 20 -d grab behave -g -rc ./assets/object_pop_up_noclass.pth -c configs/smplh.toml
# model with class prediction (assumes 24GB GPU memory)
python evaluate.py scenarios/gb_PNv2_class.toml -b 64 -w 20 -d grab behave -g -rc ./assets/object_pop_up_class.pth -c configs/smplh.toml
```


## Training
Use the following command to run the training:
```bash
python train.py scenarios/gb_PNv2_noclass.toml -b 32 -w 10 -nowb -ep 0001_smplh -c configs/smplh.toml
```


## Citation
```bibtex
@inproceedings{petrov2023popup,
   title={Object pop-up: Can we infer 3D objects and their poses from human interactions alone?},
   author={Petrov, Ilya A and Marin, Riccardo and Chibane, Julian and Pons-Moll, Gerard},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   year={2023}
}
```


## Acknowledgements
This project benefited from the following resources:
* [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/license.html) and [GRAB](https://grab.is.tue.mpg.de/) datasets;
* [grab](https://github.com/otaheri/GRAB) preprocessing code; 
* [smplx](https://github.com/vchoutas/smplx) repository: SMPL-X to SMPL+H conversion;
* [PointNet v2 implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/): encoder model;
* [blendify](https://github.com/ptrvilya/blendify/): all visualizations;
* [blogpost](http://danshiebler.com/2016-09-14-parallel-progress-bar/): parallel map implementation.
