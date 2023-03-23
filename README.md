# PIFiA: Self-supervised Approach for Protein Functional Annotation from Single-Cell Imaging Data

![Top language](https://img.shields.io/github/languages/top/arazd/pifia)
![License](https://img.shields.io/github/license/arazd/pifia)
[![Generic badge](https://img.shields.io/badge/DOI-10.1101/2023.02.24.529975-ORANGE.svg)](https://shields.io/)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.1101/zenodo.6762584.svg)](https://doi.org/10.1101/2023.02.24.529975) -->

<img align="right" src="https://github.com/arazd/pifia/blob/main/images/pifia_icon.png" alt="PIFiA" width="90"/>

We present **PIFiA** (Protein Image-based Functional Annotation), a self-supervised approach for protein functional annotation from single-cell imaging data. We imaged the global yeast ORF-GFP collection and applied PIFiA to generate protein feature profiles from single-cell images of fluorescently tagged proteins. We show that PIFiA outperforms existing approaches for molecular representation learning and describe a range of downstream analysis tasks to explore the information content of the feature profiles.

**Check out our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.02.24.529975v1)**!

## About
Despite major developments in molecular representation learning, **extracting functional information from biological images** remains a non-trivial
computational task. In this work, we revisit deep learning models used for *classifying major subcellular localizations*, and evaluate
*representations extracted from their final layers*. We show that **simple convolutional networks trained on localization classification can learn protein representations that encapsulate diverse functional information**, and significantly outperform currently used autoencoder-based models. 

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



## Installation

### System requirements


### Dependencies
Our implementation is based on Python 3 and TensorFlow 2.2. 

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
git clone https://github.com/arazd/pifia
cd pifia
conda env create -f environment.yml
```
Your conda should start downloading and extracting packages. This can take ~15-20 minutes.

To activate the environment, run:
```bash
conda activate conda_env
```
<!--
pip install tensorflow-gpu=2.2.0

conda install cudatoolkit==10.1.243
conda install cudnn==7.6.5

pip install sklearn numpy Pillow argparse matplotlib
-->
## Demo
Here we show how to run PIFiA demo on a toy dataset (5 proteins).

1. First, unzip the toy dataset folder:
```bash
cd pifia
unzip data/data_subset.zip
```
### Training PIFiA
2. Create folders for checkpointing / saving model weights:
```bash
mkdir ckpt_dir
mkdir saved_weights
```
Since our full dataset contrains >3 million single-cell images, it is expensive to run feature extraction during training. Hence, we save model weights several times during training, then perform feature extraction and evaluation, and finally select the best weights.

Checkpointing is implemented for training on high-performance computing facilities that require job preemption.

3. Run training script:
```bash
export HDD_MODELS_DIR=./ 
source activate conda_env

python model/train.py --dataset harsha  \
    --backbone pifia_network --learning_rate 0.0003 --dropout_rate 0.02 --cosine_decay True \
    --labels_type toy_dataset --dense1_size 128 --num_features 64 --save_prefix TEST_RUN
    --num_epoch 30  --checkpoint_interval 1800 --checkpoint_dir ./ckpt_dir --log_file /log_file.log
```

OR, if you are using slurm, run:
```bash
sbatch scipts/train_pifia.sh
```

After training is completed, you can see training log and saved weights in ```saved_weights``` folder we created.

### Loading pre-trained PIFiA model and feature extraction
4. To load a pre-trained PIFiA model, you can:

5. To extract single-cell features, you can:

## References 

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
