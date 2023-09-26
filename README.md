# LiveTransForM

# Requirements
* RDKit (version >= 2022.09.5)
* Python (version >= 3.7.13)
* PyTorch (version >= 1.13.0)
* Openbabel (version >= 3.1.0)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

# Quick Start

## Code for  Training model
This repository contains the Python 3 implementation of the new Fast Junction Tree Variational Autoencoder code.

* `script/` contains codes for VAE training,TL SSmodel training. Please refer to `script/README.md` for details.
* `fast_jtnn/` contains codes for model implementation.

## Usage
The SSVAE model is define in 
script/Find_ss.py
