#!/bin/bash

conda create -n aaa python=3.8
conda activate aaa
pip3 install nltk
pip3 install transformers==4.3.0
pip3 install scikit-learn==0.24.1
pip3 install packaging==21.3
# pip3 install 'torchmetrics<0.8' # NOT SURE IF I NEED THIS