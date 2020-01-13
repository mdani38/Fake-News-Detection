# Fake News Classification: Natural Language Processing of Fake News Shared on Twitter 

By Matthew Danielson 

## The Project

This project is a NLP classification effort using the FakeNewsNet dataset created by the The Data Mining and Machine Learning lab (DMML) at ASU. It is created using multiple fact checkers to create labels of fake and real news from articles shared on twitter.

The original paper on the dataset: https://arxiv.org/abs/1809.01286
Additional papers on the dataset: https://arxiv.org/abs/1712.07709 and https://arxiv.org/abs/1708.01967

**I did not initially collect this data or own the processes used to create it. My analysis and models are based on their prexisting work and data**


I create 3 models here using the fake news dataset:

1. A simple baseline classification model of fake news with Bag of Words and logistic regression: 79% accuracy
2. A pre-trained neural network using BERT for classification of fake news: 71% accuracy
3. A generative model using GPT-2 to compare new examples against the text in the dataset

## Data

The data can be located in these two sources. The kaggle dataset is the original data that I use in the EDA, Visualization and Data Combinations notebook. 
https://github.com/KaiDMML/FakeNewsNet : subfolder fakenewsnet_github
https://www.kaggle.com/mdepak/fakenewsnet : subfolder fakenewsnet_kaggle

**IMPORTANT NOTE: The full dataset is NOT present in this repository. I only have what is publicly available on kaggle and the FakeNewsNet github repository. If you wish to work through the full workflow in this repository, you MUST follow the  instructions in https://github.com/KaiDMML/FakeNewsNet to access the full dataset. This will entail following Twitter's guidelines and using their API with valid keys**

In particular, the data included here is missing:

- gossipcop_fake_updated.csv
- gossipcop_real_updated.csv
- politifact_fake_updated.csv
- politifact_real_updated.csv

The data combination notebook reads in those inputs as long as they have been accessed appropriately using the scripts in the linked github and Twitter. The following files are outputs generated from the above that are not included:

- combined_text.csv
- combined_text.json
- merged.csv
- train.csv
- test.csv

## Notebooks

The notebooks are listed here in order of usage:

1. EDA - EDA.ipynb
2. Visualization - visualizations.ipynb
3. Combining the final data after using linked github repository and accessing data from twitter - data_combination.ipynb
4. Model 1: Simple baseline using BOW and TDIF vectorization and LogReg - simple_baseline.ipynb
5. Model 2: Neural network using pretrained BERT and trained on fake news dataset for classification - bert.ipynb
6. Model 3: Pretrained GPT-2 to generate new examples for comparison against the text in the dataset

## Results 

All results and visualizations are present in below presentation.

https://www.canva.com/design/DADp9gRmr78/ggbpzPMzuxSUXoPQeIvSDw/view?utm_content=DADp9gRmr78&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton#1


## References

@article{shu2018fakenewsnet,
  title={FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media},
  author={Shu, Kai and  Mahudeswaran, Deepak and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  journal={arXiv preprint arXiv:1809.01286},
  year={2018}
}

@article{shu2017fake,
  title={Fake News Detection on Social Media: A Data Mining Perspective},
  author={Shu, Kai and Sliva, Amy and Wang, Suhang and Tang, Jiliang and Liu, Huan},
  journal={ACM SIGKDD Explorations Newsletter},
  volume={19},
  number={1},
  pages={22--36},
  year={2017},
  publisher={ACM}
}

@article{shu2017exploiting,
  title={Exploiting Tri-Relationship for Fake News Detection},
  author={Shu, Kai and Wang, Suhang and Liu, Huan},
  journal={arXiv preprint arXiv:1712.07709},
  year={2017}
}

https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
https://github.com/huggingface/transformers
https://github.com/scoutbee/pytorch-nlp-notebooks

