import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DR_COVID_RESULTS_URL = \
    'https://gist.github.com/EightSQ/c930637d143cbb7c85f2a9334e18a96b/raw/20139258e4a4848884fc7755914d2a19055b4368/drcovid_results.json'

DR_COVID_RESULTS_FILENAME = 'drcovid_results.json'

COVID_CLINICAL_TRAILS_FILENAME= "covid_trials.csv"

DRUG_REPURPOSING_DS_FILE_NAME = 'drug_repurposing.csv'

NUM_OF_COVID_VARIANTS = 33

DRUG_NAME_MAX_LEN = 30

COVID_VARIANT_PREFIX = 'COVID'

DRUG_NAME_COLUMN = 'DRUG_NAME'

LABEL_COLUMN = 'IN_CLINICAL_TRIALS'

STUDY_TYPE_COLUMN = 'study_type'

STUDY_TYPE_INTERVENTIONAL = 'Interventional'

INTERVENTION_COLUMN = 'intervention'

PARAM_GRID_SVC  = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}

PARAM_GRID_K_NEIGHBORS = {
    'n_neighbors': [2, 4, 6, 8, 10 ,12],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']
}

PARAM_GRID_DECISION_TREE =  {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'min_samples_split': [4, 6, 8, 10, 12, 15],
    'max_features': ['auto', 'sqrt', 'log2']
}

PARAM_GRID_RANDOM_FOREST = {
    'n_estimators': [200, 500, 1000, 1500],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_features': ['auto', 'sqrt', 'log2']
}

PARAM_GRID_MLP = {
    'hidden_layer_sizes':  [(10), (20,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

ESTIMATORS = {
    "SVC": {
        "estimator_object": SVC(),
        "param_grid": PARAM_GRID_SVC
    },
    # SVC
    # best_params: {'C': 1000, 'gamma': 0.001}
    # test: 0.5
    # train: 0.5

    "KNeighborsClassifier": {
        "estimator_object": KNeighborsClassifier(),
        "param_grid": PARAM_GRID_K_NEIGHBORS
    },
    # KNeighborsClassifier
    # best_params: {'algorithm': 'ball_tree', 'n_neighbors': 12, 'weights': 'uniform'}
    # test: 0.5
    # train: 0.5075

    "DecisionTreeClassifier": {
        "estimator_object": DecisionTreeClassifier(),
        "param_grid": PARAM_GRID_DECISION_TREE
    },
    # DecisionTreeClassifier
    # best_params: {'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_split': 10, 'splitter': 'random'}
    # test: 0.5484779116465864
    # train: 0.687865394274234

    "RandomForestClassifier": {
        "estimator_object": RandomForestClassifier(),
        "param_grid": PARAM_GRID_RANDOM_FOREST
    },
    # RandomForestClassifier
    # best_params: {'criterion': 'log_loss', 'max_features': 'sqrt', 'n_estimators': 1500}
    # test: 0.52
    # train: 1.0

    "MLPClassifier": {
        "estimator_object": MLPClassifier(),
        "param_grid": PARAM_GRID_MLP
    }
    # MLPClassifier
    # best_params: {'activation': 'tanh', 'hidden_layer_sizes': (20,), 'learning_rate': 'invscaling', 'solver': 'adam'}
    # test: 0.52
    # train: 0.5323744349573079
}

SELECTED_ESTIMATOR_NAME = "DecisionTreeClassifier" # can be changed to other model class in the future accroding to changes in grid-search results

SCORING_METRIC = 'roc_auc'

NUMBER_OF_FOLDS = 10

class DataCannotBeLoadedException(Exception):
    def __init__(self):
        super(DataCannotBeLoadedException, self).__init__("Data cannot be loaded")

def load_dataset(path):
    data = pd.read_csv(path, sep=',')

    return data
