# Drug repurposing for the SARS-CoV-2 pandemic -
predicting whether an existing drug will be tested in clinical trials

## Description

The SARS-CoV-2 pandemic caused more than 769 million cases and
6.9 million deaths worldwide. The development of completely new drugs for
such a novel disease is a challenging, time-intensive process. This emphasizes the
importance of drug repurposing, where treatments are found among existing drugs
meant for different diseases. A promising approach to this is based on combining
knowledge graphs with state-of-the-art results from graph neural networks. So
far, such approaches only considered the unsupervised setting. However, since the
outbreak of SARS-CoV-2 a few years ago, several clinical trials have already been
conducted on multiple drugs. In this work, we revisit the established DR-COVID
model and add supervision of the lists of clinical trials that were conducted.

## Getting Started

### Executing program

#### Phase 1

The main entry to the program is in the file ```drug_repurposing.py```

The main function is ```DrugRepurposing().run()```

The function is the main entry point which do the following main actions:
  
* build a data-set
* print some debug information
* run the model:
  * load the data-set created
  * split to train and test
  * get best params by grid search
  * train the model
  * predict (train and test sets)
  * get score according to the metric selected

the ```config.py``` file contains serverl configuration values related to the the data sources, learning models and grid search hyper parameters.

In order to test some other models and thier results you can change the following config values :
* SELECTED_ESTIMATOR_NAME
* SCORING_METRIC
* NUMBER_OF_FOLDS

#### Phase 2

We executed Automated Machine Learning by using mljar-supervised  Automated Machine Learning Python package that works with tabular data. It abstracts the common way to preprocess the data, construct the machine learning models, and perform hyper-parameters tuning to find the best model. The mljar-supervised help us with:
explaining and understanding our data, trying many different machine learning models,
creating Markdown reports from analysis with details about all models,
saving, re-running and loading the analysis and Machine Learning models.

Result analysis can be found in : https://drive.google.com/drive/folders/12139mlIR8cagEUBiZwLRTq-PWjMWA0A0

## Authors

Kfir Avlas
kfiravlas@gmail.com

## Version History

* 0.1
    * Initial Release
* 0.2      
    * Automated Machine Learning
