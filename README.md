# Adversifier

Official repository for the Adversarial Attacks against Abuse (AAA) benchmark. AAA is a benchmark for evaluating abuse detection systems while avoiding rewarding them for being able to model the bias in the datasets.

## Setup

In order to run the AAA benchmark, you need to download the ORG dataset. The ids of the tweets included in the dataset are listed at [org_ids.tsv](https://drive.google.com/file/d/1RpzRPGCQxhuTchSojHpFL9-4xaMFgH1U/view?usp=sharing). After downloading the tweets, put them into a tab-separated file (_DATA/org.tsv_). Each line should contain the tweet id followed by the content of the tweet.

## Supported Models

The weights of **BERT_KEN** [(Kennedy et al., 2020)]() can be downloaded at:
* [sexism.bin](https://drive.google.com/file/d/1qVsRTEFUPYWEKuY4gEsU2qSKk28vIagl/view?usp=sharing)
* [racism.bin](https://drive.google.com/file/d/1waS2kcmw3ayEonK9fofsTKweqYOdguJu/view?usp=sharing)

Put both files in the _models_ directory.

## How to evaluate your own model