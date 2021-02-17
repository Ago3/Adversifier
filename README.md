# Adversifier

Official repository for the Adversarial Attacks against Abuse (AAA) evaluation tool. AAA is a new evaluation metric for abuse detection systems that better captures a model's performance on certain classes of hard-to-classify microposts, and for example penalises systems which are biased on low-level lexical features.

## Setup
<!-- Add other requirements: nltk, sklearn -->
<!-- In order to run the AAA benchmark, you need to download the ORG dataset. The ids of the tweets included in the dataset are listed in _DATA/org_ids.tsv_. After downloading the tweets, put them into a tab-separated file (_DATA/org.tsv_). Each line should contain the tweet id followed by the content of the tweet. -->
If willing to replicate our results with the BERT<sub>MOZ</sub> or BERT<sub>KEN</sub> models, you'll need to install the [transformers](https://huggingface.co/transformers/) library:
```
pip3 install transformers
```
All the files' paths (e.g., data files, models' checkpoints) are specified within the _info/info.py_ file. Customise this file to meet your needs.

## Datasets
For the AAA tool to run, you'll need to provide both a training and test set. Both sets should be in the form:
```
data_split = [list of posts, list of labels, list of any extra information your model might use]
```
Therefore,  the i<sup>th</sup> element of each list will contain information regarding the i<sup>th</sup> instance in the split.
Labels are assumed to be binary, with 1 corresponding to the abusive class, and 0 to the non-abusive class.

### Waseem et al., 2018 ###
<!-- To replicate our experiments on the [Waseem et al., 2018](https://link.springer.com/chapter/10.1007/978-3-319-78583-7_3)'s dataset,  -->

### Davidson et al., 2017 ###
To run the AAA tool on the [Davidson et al., 2017](https://ojs.aaai.org/index.php/ICWSM/article/view/14955)'s dataset, download the [_davidson_data.csv_](https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv) file and add it to the _DATA_ directory. You can then call the _utils.get_davidson_data_ function, that returns a dictionary with keys {'train', 'test'} and the corresponding data_split as argument.<br/>
Splits are created using stratified sampling to split 0.8, 0.1, and 0.1 portions of tweets from each class into training, validation and test sets. The corresponding ids can be found in the _davidson_train_ids.csv_, _davidson_val_ids.csv_ and _davidson_test_ids.csv_ files within the _DATA_ directory.<br/>
Note that the _utils.get_davidson_data_ function maps the "hate speech" and "offensive" labels into the abusive class, and the "neither" label into the non abusive class.

## Supported Models
We provide code and checkpoints for the SVM, BERT<sub>MOZ</sub> and BERT<sub>KEN</sub> models trained on the Waseem et al., 2018 and Davidson et al., 2017 datasets.

### Waseem et al., 2018 ###
To replicate our experiments on the [Waseem et al., 2018](https://link.springer.com/chapter/10.1007/978-3-319-78583-7_3)'s dataset you'll need to download the following checkpoints. Add all the files to the _models_ directory, or modify the _info/info.py_ file accordingly.

#### SVM ####
The weights of our SVM model can be downloaded at:
* [sexism_model.pkl]()
* [sexism_vectorizer.pkl]()
* [racism_model.pkl]()
* [racism_vectorizer.pkl]()

#### BERT<sub>MOZ</sub> ####
The weights of our re-implementation of BERT<sub>MOZ</sub> [(Mozafari et al., 2019)]() can be downloaded at:
* [mozafari_waseem.pt]()
* [mozafari_waseem_nh.pt]() (variant of the BERT<sub>MOZ</sub> model that fully discards hashtag content)

#### BERT<sub>KEN</sub> ####
The weights of BERT<sub>KEN</sub> [(Kennedy et al., 2020)](https://arxiv.org/pdf/2005.02439.pdf) can be downloaded at:
* [sexism.bin]()
* [racism.bin]()

### Davidson et al., 2017 ###
To replicate our experiments on the [Davidson et al., 2017](https://ojs.aaai.org/index.php/ICWSM/article/view/14955)'s dataset you'll need to download the following checkpoints. Add all the files to the _models_ directory, or modify the _info/info.py_ file accordingly.

#### SVM ####
The weights of our SVM model can be downloaded at:
* [hate_speech_model.pkl]()
* [hate_speech_vectorizer.pkl]()
* [offensive_model.pkl]()
* [offensive_vectorizer.pkl]()

#### BERT<sub>MOZ</sub> ####
The weights of our re-implementation of BERT<sub>MOZ</sub> [(Mozafari et al., 2019)]() can be downloaded at:
* [mozafari_davidson.pt]()

#### BERT<sub>KEN</sub> ####
The weights of BERT<sub>KEN</sub> [(Kennedy et al., 2020)](https://arxiv.org/pdf/2005.02439.pdf) can be downloaded at:
* [hate_speech.bin]()
* [offensive.bin]()

## Computing the AAA score for the supported models
To replicate the experiments reported in the AAA paper, download the data files and models' checkpoints as described above, and run the following command:
```
python3 main.py
```

## How to evaluate your model on a dataset
To run the AAA tool on your model with a generic dataset, you'll need to provide:
* the training and test sets, in the format specified [here](#Datasets "Goto Datasets").
* your model's predictor: a function that takes as input a list of arguments, the 1<sup>st</sup> one being a list of *NON-pre-processed* posts, and returns a list of binary predictions.<br/>
Here is an example:
```
from AAAdversifier import AAAdversifier


adversifier = AAAdversifier()
train_data, test_data = load_your_data()
adversifier.aaa('your_model_name', your_model.predictor, train_data, test_data)
```
Check _main.py_ for usage examples.
