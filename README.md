# Adversifier

Official repository for the Adversarial Attacks against Abuse (AAA) evaluation tool. AAA is a new evaluation metric that better captures a model's performance on certain classes of hard-to-classify microposts, and for example penalises systems which are biased on low-level lexical features.

## Setup
<!-- Add other requirements: nltk, sklearn -->
<!-- In order to run the AAA benchmark, you need to download the ORG dataset. The ids of the tweets included in the dataset are listed in _DATA/org_ids.tsv_. After downloading the tweets, put them into a tab-separated file (_DATA/org.tsv_). Each line should contain the tweet id followed by the content of the tweet. -->
If willing to replicate our results with the BERT<sub>MOZ</sub> or BERT<sub>KEN</sub> models, you'll need to install the [transformers](https://huggingface.co/transformers/) library:
```
pip3 install transformers
```
All the files' paths (e.g., data files, models' checkpoints) are specified within the _info/info.py_ file. Customise the file to meet your needs.

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
To run the AAA tool on the [Davidson et al., 2017](https://ojs.aaai.org/index.php/ICWSM/article/view/14955)'s dataset, download the [_davidson_data.csv_](https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv) file and add it to the _DATA_ directory. You can then call the _utils.get_davidson_data_ function, that returns a dictionary with keys {'train', 'test'} and the corresponding data_split as argument.
Note that this function maps the "hate speech" and "offensive" labels into the abusive class, and the "neither" label into the non abusive class.

## Supported Models

The weights of **BERT<sub>KEN</sub>** [(Kennedy et al., 2020)](https://arxiv.org/pdf/2005.02439.pdf) can be downloaded at:
* [sexism.bin](https://drive.google.com/file/d/1qVsRTEFUPYWEKuY4gEsU2qSKk28vIagl/view?usp=sharing)
* [racism.bin](https://drive.google.com/file/d/1waS2kcmw3ayEonK9fofsTKweqYOdguJu/view?usp=sharing)

Put both files in the _models_ directory.

## How to evaluate your own model