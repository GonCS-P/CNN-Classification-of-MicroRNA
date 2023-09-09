# Convolution-Neural-Network-based Single-molecule Classification of Circulating MicroRNA Mixtures

**Author**: Gon Eyal
**Supervisor**: Dr. Yohai Bar Sinai
**Research Team**: Amit Federbush, Jonathan Jeffet, Nadav Tenenboim
**Assisted by**: Prof. Yuval Ebenstein
**Date**: 30/08/2023

## Abstract

MicroRNA (miR) represents a category of small non-coding RNAs in the regulation of gene expression, and they are gaining recognition as potential indicators of diseases, such as cancer. These miRs can be found in the blood plasma via liquid biopsy analysis. However, extracting information from a single molecule microscopy measurement remains a major scientific challenge. This study presents a new technique to detect and classify multiplexed single-molecule of a selected panel of miRs. The study's procedure (pipeline) maximizes accuracy by encompassing data processing and a suitable classification technique, achieved through an exploration of various Machine Learning (ML) models. The proposed technique relies on a Convolution Neural Net (CNN) Machine Learning-centered detection approach, implemented on convoluted labeled imagery data of miRs. This technique significantly improves classification accuracy in terms of recall and precision compared to earlier methods and has greater generalizability to a broader spectrum of miR types.

## Methods

### Data Processing

We utilized ImageJ, a versatile image analysis software, to quantify microscopy data. This software facilitated manual color classification to create ground truth (GT) data. ImageJ's extensions, including ThunderSTORM, enabled single molecule detection using techniques like PALM and STORM.

Our pipeline involved basic linear transformations to normalize raw samples and ThunderSTORM blob detection. These blobs were manually categorized into 3 miR types and 1 noise class.

We explored two techniques for processing normalized samples:
- **Cross-correlation**
- **Cosine similarity**

### Model Architecture

In machine learning, data division into training, validation, and test sets significantly impacts model performance. We partitioned our dataset (655 red, 141 green, 611 blue miRs, and 2629 noise instances) into 60% training, 20% validation, and 20% test sets. We selected the LeNet-5 Convolutional Neural Network (CNN) for image classification due to its proven ability to extract features effectively from visual data.

## Content

- **Research Summary.pdf** The study output paper.
- **Cross-Correlation-V3-Cleaned.ipynb** Consists of the study's code
- **masks + images** Folder containing the produced miR types tamplates


## Getting Started

To replicate the results and run the code provided in this repository, follow the steps below:

1. Navigate to the repository's directory: cd your-repo
2. Create a Conda environment: conda activate ml_cocos_env
3. You can now run a Jupyter Notebook to interact with the code and perform your own experiments: jupyter notebook
