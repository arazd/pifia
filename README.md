# PIFiA: Self-supervised Approach for Protein Functional Annotation from Single-Cell Imaging Data

![Top language](https://img.shields.io/github/languages/top/arazd/pifia)
![License](https://img.shields.io/github/license/arazd/pifia)

In this work, we explored different methods for protein representation learning from microscopy data. We evaluated the extracted representations on four biological benchmarks - subcellular compartments, biological processes, pathways and protein complexes.
<img src="https://github.com/arazd/pifia/blob/main/images/pifia_icon.png" alt="PIFiA" width="70"/>

**Check out our [biorXiv preprint](https://www.biorxiv.org/content/10.1101/2023.02.24.529975v1)**!

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



## Codebase
Configure environment:
```bash
pip install tensorflow-gpu=2.2.0

conda install cudatoolkit==10.1.243
conda install cudnn==7.6.5

pip install sklearn numpy Pillow argparse matplotlib
```

Code for DeepLoc and Paired Cell Inpainting models is based on TensorFlow 2 and available under ```models/keras_models.py```. 

To run train script, use:
```python train.py --backbone deep_loc --learning_rate 0.001 --num_epoch 5 --checkpoint_dir save_ckpt```

Check other ```train.py``` arguments for different training options.

CellProfiler features were obtained by running the [CellProfiler pipeline](https://cellprofiler.org/). 

Dataloader class is available under ```models/dataset_utils.py```, and contains reading / preprocessing / converting to tensorflow Dataset single-cell images in .tiff format. Link to server with cell image data is [above](#server).


## References 

If you found this work useful for your research, please cite:

Razdaibiedina A, Brechalov A. Learning multi-scale functional representations of proteins from single-cell microscopy data. InICLR2022 Machine Learning for Drug Discovery 2022 Mar 31.

```
@article{razdaibiedina2022learning,
  title={Learning multi-scale functional representations of proteins from single-cell microscopy data},
  author={Razdaibiedina, Anastasia and Brechalov, Alexander},
  journal={arXiv preprint arXiv:2205.11676},
  year={2022}
}
```

<!--The supervised model we used for representation learning was first introduced in this paper:

Kraus OZ, Grys BT, Ba J, Chong Y, Frey BJ, Boone C, Andrews BJ. Automated analysis of high‐content microscopy data with deep learning. Molecular systems biology. 2017 Apr;13(4):924.

```
@article{kraus2017automated,
  title={Automated analysis of high-content microscopy data with deep learning},
  author={Kraus, Oren Z and Grys, Ben T and Ba, Jimmy and Chong, Yolanda and Frey, Brendan J and Boone, Charles and Andrews, Brenda J},
  journal={Molecular systems biology},
  volume={13},
  number={4},
  pages={924},
  year={2017}
}
```
-->
