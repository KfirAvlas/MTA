# Drug repurposing for the SARS-CoV-2 pandemic -
predicting whether an existing drug will be tested in clinical trials

## Description

The active global SARS-CoV-2 pandemic caused more millions cases and deaths worldwide. The development of completely new drugs for such a novel disease is a challenging, time intensive process. 
This emphasizes the importance of drug repurposing, where treatments are found among existing drugs that are meant for different diseases.
In the following project I was trying to predict whether an existing drug will be tested in COVID-19-related clinical trials. 
I was using a machine learning algorithm to train a classifier model and predict whether an unseen drug will be tested in COVID-19-related clinical trials. 
I was trying to use several ML classifiers in order to achieve the best model for prediction. 

## Getting Started

### Executing program

The main entry to the program is in the file ```drug_repurposing.py```

The mail function is ```DrugRepurposing().run()```

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


## Authors

Kfir Avlas
kfiravlas@gmail.com

## Version History

* 0.1
    * Initial Release
