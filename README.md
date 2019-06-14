# (Semi)-Unsupervised clustering of drosophila behaviour
## EPFL Spring Semester 2019, Ramdya Lab, Neuroengineering Lab, Semester Project

Code for my semester project on drosophila behaviour. The overall goal was to define a model which
can identify new clusters, using the latent-space of an VAE. For this multiple models were designed,
trained and evaluated. Check out `reparam_vae.ipynb` for an overview, and main entrypoint.

Also `drosoph_vae.settings.config.py` is also very important, you'll define how the data is loaded,
hyperparameters, general settings, and much more there.

Sources and inspiration are noted directly next to the applicable source code.

This project uses Tensorflow 2.0 features, most prominent Eager-mode. So keep that in mind.

## Module overview


```
├── archive                        # General archive, models no longer supported, playground, ...
├── LICENSE
├── README.md                      # This document
├── reparam_vae.ipynb              # Main entrypoint and training notebook
│                                  # If you run this a lot of stuff will be saved to disk.
├── reparam_vae.py                 # Python version of the notebook
└── drosoph_vae
    ├── data_loading.py            # Main entrypoint to load the data
    ├── helpers                    # Various general helpers and utils
    │   ├── file.py
    │   ├── jupyter.py
    │   ├── logging.py
    │   ├── misc.py
    │   ├── plots.py
    │   ├── tensorflow.py
    │   └── video.py
    ├── layers                     # Special layers, some used layers are defined directly
    │   │                          # alongside the models (not ready to be used somewhere else)
    │   ├── padded_conv1d_transposed.py
    │   ├── temporal_block.py
    │   └── temporal_upsampling_conv.py
    ├── losses                     # Loss computation, and metrics
    │   ├── normalized_mutual_information.py
    │   ├── purity.py
    │   ├── triplet_loss.py
    │   └── vae_loss.py
    ├── models                     # Models! The magic happens here.
    │   ├── drosoph_vae_conv2d.py  # WIP (doesn't work right now)
    │   ├── drosoph_vae_conv.py
    │   ├── drosoph_vae.py
    │   ├── drosoph_vae_skip_conv.py
    │   └── utils.py
    ├── preprocessing.py           # To be consumed with `data_loading.py`
    ├── settings                   # Second most interesting place. Check out the config
    │   ├── config.py              # <-------------------------------- (it's quite cool)
    │   ├── data.py
    │   └── skeleton.py            # Stolen from another repo in the group
    └── training                   # Training procedures. Check out `reparam_vae.ipynb` and it will
        │                          # become clearer.
        ├── supervised.py
        ├── utils.py
        └── vae.py
```

## Installation of tf-nightly gpu


**Basic installation**: (that you might want to adapt)

The official installation guides on Tensorflow GPU are quite nice. Just make sure you purge all
nvidia-drivers prior to installation. 

```
pip install --upgrade tf-nightly==1.14.1-dev20190312 tf-nightly-gpu==1.14.1-dev20190312 tfp-nightly==0.7.0.dev20190312

# This should also do the trick
# pip install --upgrade tf-nightly tf-nightly-gpu tfp-nightly
```

If error `libcublas.so.10.0: cannot open shared object file: No such file or directory` occurs:

```
conda install cudatoolkit
conda install cudnn
```

If they still persist, check out the `LD_*` options and append them accordingly.

## Data copying (just for me)

```
# Use this to retrieve the images and the positional data
# You might want to adapt it to certain flys as well
find <base path to experiment>/180920_aDN_CsCh -name "*.jpg" -o -name "*.pkl" | grep -E "camera_1_img|pose" | xargs cp -t /target/path --parent
```

```
rsync -azvh ramdya_lab:/target/path/maybe/without/some/of/the/parents /local/path
```
