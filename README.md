# Adversifier

Official repository for the Adversarial Attacks against Abuse (AAA) evaluation tool. AAA is a new evaluation metric for abuse detection systems that better captures a model's performance on certain classes of hard-to-classify microposts, and for example penalises systems which are biased on low-level lexical features.

## Setup
Within the _Adversifier_ directory run the following command:
```
./setup.sh
```
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
To run the AAA tool on the [Waseem et al., 2018](https://link.springer.com/chapter/10.1007/978-3-319-78583-7_3)'s dataset, download the tweets through the Twitter API and put them in _DATA/waseem_data.tsv_. The tab-separated file should have the following header (and format):
```
tweet_id	tweet_text	label
```
You can then call the _utils.get_waseem_data_ function, that returns a dictionary with keys {'train', 'test'} and the corresponding data_split as argument.<br/>
Splits are created using stratified sampling to split 0.8, 0.1, and 0.1 portions of tweets from each class into training, validation and test sets. The corresponding ids can be found in the _waseem_train_ids.csv_, _waseem_val_ids.csv_ and _waseem_test_ids.csv_ files within the _DATA_ directory.<br/>
Note that the _utils.get_waseem_data_ function maps the "sexism", "racism" and "both" labels into the abusive class, and the "neither" label into the non abusive class.

### Davidson et al., 2017 ###
To run the AAA tool on the [Davidson et al., 2017](https://ojs.aaai.org/index.php/ICWSM/article/view/14955)'s dataset, download the [_davidson_data.csv_](https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv) file and add it to the _DATA_ directory. You can then call the _utils.get_davidson_data_ function, that returns a dictionary with keys {'train', 'test'} and the corresponding data_split as argument.<br/>
Splits are created using stratified sampling to split 0.8, 0.1, and 0.1 portions of tweets from each class into training, validation and test sets. The corresponding ids can be found in the _davidson_train_ids.csv_, _davidson_val_ids.csv_ and _davidson_test_ids.csv_ files within the _DATA_ directory.<br/>
Note that the _utils.get_davidson_data_ function maps the "hate speech" and "offensive" labels into the abusive class, and the "neither" label into the non abusive class.

## Supported Models
We provide code and checkpoints for the SVM, BERT<sub>MOZ</sub> and BERT<sub>KEN</sub> models trained on the Waseem et al., 2018 and Davidson et al., 2017 datasets.

### Waseem et al., 2018 ###
To replicate our experiments on the Waseem et al., 2018's dataset you'll need to download the following checkpoints. **You can download all the checkpoints from [here](https://drive.google.com/file/d/1N6J67yGOVKZTphVPteWGIS_vDqDQq_g_/view?usp=sharing)** (3.01 GB), or download the ones of interest from the following list. Add all the files to the _models_ directory, or modify the _info/info.py_ file accordingly.

#### SVM ####
The weights of our SVM model can be downloaded at:
* [sexism_model.pkl](https://drive.google.com/file/d/19uVCQm0o5IHOI3jlJ8EM8Bw1dAI1Fy91/view?usp=sharing)
* [sexism_vectorizer.pkl](https://drive.google.com/file/d/1LV5_KL-neQkm3sKGwjd3pIzPLk_yiP8h/view?usp=sharing)
* [racism_model.pkl](https://drive.google.com/file/d/1vRqbuqXSUnqRK2ruL1a2WIDGVlcJh5QI/view?usp=sharing)
* [racism_vectorizer.pkl](https://drive.google.com/file/d/1FS9vyHtjOUbSeXROt33RchDL13ZQsAeE/view?usp=sharing)

#### BERT<sub>MOZ</sub> ####
The weights of our re-implementation of BERT<sub>MOZ</sub> [(Mozafari et al., 2019)](https://arxiv.org/pdf/1910.12574.pdf) can be downloaded at:
* [mozafari_waseem.pt](https://drive.google.com/file/d/1LyJAy74RzqGe2Hg-INZOjlXhEnDsTGWP/view?usp=sharing)
* [mozafari_waseem_nh.pt](https://drive.google.com/file/d/1-tbY0IOzjvbcu2utZ4RF1biAiUpXiHpU/view?usp=sharing) (variant of the BERT<sub>MOZ</sub> model that fully discards hashtag content)

#### BERT<sub>KEN</sub> ####
The weights of BERT<sub>KEN</sub> [(Kennedy et al., 2020)](https://arxiv.org/pdf/2005.02439.pdf) can be downloaded at:
* [sexism.bin](https://drive.google.com/file/d/1F0N0FZSBSkdm4EEGnH8mbB0m6FBDg4fj/view?usp=sharing)
* [racism.bin](https://drive.google.com/file/d/1TbWGI0142DpN4shmLctOlDlK0fY42-tU/view?usp=sharing)

### Davidson et al., 2017 ###
To replicate our experiments on the Davidson et al., 2017's dataset you'll need to download the following checkpoints. **You can download all the checkpoints from [here](https://drive.google.com/file/d/1O6q67BLD-q531odcu1grH2ioCY7OjDV1/view?usp=sharing)**, or download the ones of interest from the following list. Add all the files to the _models_ directory, or modify the _info/info.py_ file accordingly.

#### SVM ####
The weights of our SVM model can be downloaded at:
* [hate_speech_model.pkl](https://drive.google.com/file/d/1MPpb-6TouSlkRJ0GkeYIwkG2R-UONZze/view?usp=sharing)
* [hate_speech_vectorizer.pkl](https://drive.google.com/file/d/1g9clFa9fENLjumFrTE7IMT849n5NmKjR/view?usp=sharing)
* [offensive_model.pkl](https://drive.google.com/file/d/15QvP5EGffUAwtkwSwfjJRrunnpwqdmNc/view?usp=sharing)
* [offensive_vectorizer.pkl](https://drive.google.com/file/d/1lqsNOTT7ZwIEgPClcrWeFN4j5WMeHbrr/view?usp=sharing)

#### BERT<sub>MOZ</sub> ####
The weights of our re-implementation of BERT<sub>MOZ</sub> (Mozafari et al., 2019) can be downloaded at:
* [mozafari_davidson.pt](https://drive.google.com/file/d/1FFspZaUiznGKpqBtaOTseqSF-ple5KOs/view?usp=sharing)

#### BERT<sub>KEN</sub> ####
The weights of BERT<sub>KEN</sub> (Kennedy et al., 2020) can be downloaded at:
* [hate_speech.bin](https://drive.google.com/file/d/17_AInLbhhx9M7I1ldFcrGOGxoNeXDAXa/view?usp=sharing)
* [offensive.bin](https://drive.google.com/file/d/1JsamtJ8Xa27tG4yG_o6ufLSwTCKDcmTu/view?usp=sharing)

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
