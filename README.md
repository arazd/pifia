# PIFiA: Self-supervised Approach for Protein Functional Annotation from Single-Cell Imaging Data

![Top language](https://img.shields.io/github/languages/top/arazd/pifia)
![License](https://img.shields.io/github/license/arazd/pifia)

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
Configure environment:
```bash
git clone https://github.com/arazd/pifia
cd pifia
conda env create -f environment.yml
conda activate conda_env
```
<!--
pip install tensorflow-gpu=2.2.0

conda install cudatoolkit==10.1.243
conda install cudnn==7.6.5

pip install sklearn numpy Pillow argparse matplotlib
-->
## Demo
To run PIFiA demo on a toy dataset (5 proteins), first unzip the toy dataset folder:
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

### Loading pre-trained PIFiA model and feature extraction


3. 

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
