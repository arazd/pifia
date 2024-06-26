# PIFiA: Self-supervised Approach for Protein Functional Annotation from Single-Cell Imaging Data

![Top language](https://img.shields.io/github/languages/top/arazd/pifia)
![License](https://img.shields.io/github/license/arazd/pifia)
[![Generic badge](https://img.shields.io/badge/DOI-10.1101/2023.02.24.529975-ORANGE.svg)](https://doi.org/10.1101/2023.02.24.529975)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.1101/zenodo.6762584.svg)](https://doi.org/10.1101/2023.02.24.529975) -->

<img align="right" src="https://github.com/arazd/pifia/blob/main/images/pifia_icon.png" alt="PIFiA" width="90"/>

We present **PIFiA** (Protein Image-based Functional Annotation), a self-supervised approach for protein functional annotation from single-cell imaging data. 

**Check out our [Molecular Systems Biology article](https://www.embopress.org/doi/full/10.1038/s44320-024-00029-6)**!

## About
We imaged the global yeast ORF-GFP collection and applied PIFiA to generate protein feature profiles from single-cell images of fluorescently tagged proteins. We show that PIFiA outperforms existing approaches for molecular representation learning and describe a range of downstream analysis tasks to explore the information content of the feature profiles.

<!-- Despite major developments in molecular representation learning, **extracting functional information from biological images** remains a non-trivial
computational task. In this work, we revisit deep learning models used for *classifying major subcellular localizations*, and evaluate
*representations extracted from their final layers*. We show that **simple convolutional networks trained on localization classification can learn protein representations that encapsulate diverse functional information**, and significantly outperform currently used autoencoder-based models.  -->
<!-- 
## Methods & Results
We compare three methods for molecular representation learning:

* **Deep Loc** - a supervised convolutional network trained to classify subcellular localizations from images;
* **Paired Cell Inpainting** - autoencoder-based method for protein representation learning;
* **CellProfiler** - a classic feature extractor for cellular data;

We train Deep Loc and Paired Cell Inpainting models on single-cell yeast microscopy data, containing ~4K fluorescently-labeled proteins. Image data can be downloaded as zip files from this <span id="server">webserver</span>: [http://hershey.csb.utoronto.ca/image_screens/WT2/](http://hershey.csb.utoronto.ca/image_screens/WT2/)

We use 4 standards for comparison:
* [GO Cellular Component](http://geneontology.org/) (GO CC)
* [GO Biological Process)](http://geneontology.org/) (GO BP)
* [KEGG Pathways](https://www.genome.jp/kegg/pathway.html)
* [EMBL Protein Complexes](https://www.ebi.ac.uk/complexportal/home)
 -->


## Installation

### System requirements
Linux, Mac OS, Windows are supported for running the code on CPU; We recommend Linux for running experiments with GPU. At least 16GB of RAM is required to run the software. The codebase has been heavily tested on Linux 4.15.0-206-generic #217-Ubuntu.

### Dependencies
Our implementation is based on Python 3 and TensorFlow 2.1. 

Requirements:
* python 3.7
* tensorflow 2.1
* pandas 0.25.1
* numpy 1.18.1
* matplotlib 3.0.3
* seaborn 0.11.1
* pillow 6.1.0
* plotly 4.14.3

### Instructions
Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://docs.conda.io/en/latest/miniconda.html).

To run PIFiA code on GPU, make sure that you have a CUDA capable GPU and the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date. In our implementation, we used and CUDA 11.0.

Now you can configure conda environment:
```bash
$ git clone https://github.com/arazd/pifia
$ cd pifia
$ conda env create -f environment.yml
```
Your conda should start downloading and extracting packages. This can take ~15-20 minutes.

To activate the environment, run:
```bash
$ conda activate pifia_env
```
<!--
pip install tensorflow-gpu=2.2.0

conda install cudatoolkit==10.1.243
conda install cudnn==7.6.5

pip install sklearn numpy Pillow argparse matplotlib
-->
## Demo
Here we show how to run PIFiA demo on a toy dataset (5 proteins).

First, unzip the toy dataset folder:
```bash
$ cd pifia/data
$ unzip data_subset.zip
```
### A. Training PIFiA
**A.1 Create folders for checkpointing / saving model weights**:
```bash
$ cd ../
$ mkdir ckpt_dir
$ mkdir saved_weights
```
Since our full dataset contrains >3 million single-cell images, it is expensive to run feature extraction during training. Hence, we save model weights several times during training, then perform feature extraction and evaluation, and finally select the best weights.

Checkpointing is implemented for training on high-performance computing facilities that require job preemption.

**A.2 Run training script**:
```bash
$ export HDD_MODELS_DIR=./ 
$ conda activate pifia_env

$ python model/train.py --dataset harsha \
    --backbone pifia_network --learning_rate 0.0003 --dropout_rate 0.02 --cosine_decay True \
    --labels_type toy_dataset --dense1_size 128 --num_features 64 --save_prefix TEST_RUN \
    --num_epoch 30  --checkpoint_interval 1800 --checkpoint_dir ./ckpt_dir \ 
    --saved_weights_dir ./saved_weights --log_file ./log_file.log
```

OR, if you are using slurm, run:
```bash
$ sbatch scipts/train_pifia.sh
```

After training is completed, you can see training log and saved weights in ```saved_weights``` folder we created.

### B. Loading pre-trained PIFiA model and feature extraction
Loading weights for PIFiA model is very straightforward. Final pre-trained weights of PIFiA network (that we used in our paper) are stored under ```model/pretrained_weights```. Alternatively, if you are training PIFiA network from scratch (as shown in step A), your weights with epoch number should be saved in ```saved_weights``` folder.

**B.1 Load a pre-trained PIFiA model**

<!-- We show how to load pre-trained PIFiA weights (that are used in paper).  -->
First, activate your conda environment and go to ```model``` folder.
```bash
$ conda activate pifia_env
$ python
```

To load pre-trained PIFiA weights in Python, run the following code in the python shell:
```python
# import packages
>>> import numpy as np
>>> from model import models
>>> from model.extract_features import *

# we need to know number of proteins in our training data to get number of nodes for PIFiA classification layer
>>> labels_dict = np.load('data/protein_to_files_dict_toy_dataset.npy',allow_pickle=True)[()]
>>> num_classes = 4049

>>> model = models.pifia_network(num_classes,
                             k=1,
                             num_features=64,
                             dense1_size=128,
                             last_block=True)
>>> model.load_weights('model/pretrained_weights/pifia_weights_i0')
```
Note that if you want to load custom PIFiA weights (from training in step A), you need to change the weights path. Loading the model should take ~1min.

**B2. Extract single-cell features**

After loading the model, here is an example of extracting features from *NUP2* protein from our toy dataset:
```python
>>> protein_name = 'NUP2'
>>> protein_features, protein_images = get_features_from_protein(protein_name, labels_dict, model, 
                                                             average=False, subset='test')
```
In this example, ```protein_features``` is a numpy array of size ```(10, 64)```, i.e. it contains ten 64-dimensional single-cell feature profiles. The extraction process should take ~1sec on GPU and ~20sec on CPU.

## Reproducing results
To reproduce results from the paper, please check out folder ```pifia_results```. 

To interactively look at PIFiA average feature profiles, or aFPs (Fig. 3 in the paper), download file ```PIFiA_aFPs_tSNE.html``` and open it in your web browser. You should be able to see a whole-proteome tSNE with protein names and their annotations (subcellular localization, GO bioprocess, GO molecular function, pathway, protein complex).

## Cite this work

If you found this work useful for your research, please cite:

Razdaibiedina, A., Brechalov, A.V., Friesen, H., Mattiazzi Usaj, M., Masinas, M.P.D., Garadi Suresh, H., Wang, K., Boone, C., Ba, J. and Andrews, B.J., 2023. PIFiA: Self-supervised Approach for Protein Functional Annotation from Single-Cell Imaging Data. 

```
@article{razdaibiedina2023pifia,
  title={PIFiA: Self-supervised Approach for Protein Functional Annotation from Single-Cell Imaging Data},
  author={Razdaibiedina, Anastasia and Brechalov, Alexander V and Friesen, Helena and Mattiazzi Usaj, Mojca and Masinas, Myra Paz David and Garadi Suresh, Harsha and Wang, Kyle and Boone, Charlie and Ba, Jimmy and Andrews, Brenda J},
  journal={bioRxiv},
  pages={2023--02},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
