import datetime
import urllib.request
from typing import Dict


import json
import pandas as pd


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from Proj.config import DR_COVID_RESULTS_URL, DR_COVID_RESULTS_FILENAME, COVID_CLINICAL_TRAILS_FILENAME, DataCannotBeLoadedException, \
    DRUG_REPURPOSING_DS_FILE_NAME, STUDY_TYPE_COLUMN, STUDY_TYPE_INTERVENTIONAL, INTERVENTION_COLUMN, DRUG_NAME_COLUMN, \
    COVID_VARIANT_PREFIX, LABEL_COLUMN, ESTIMATORS, DRUG_NAME_MAX_LEN, load_dataset, SELECTED_ESTIMATOR_NAME, SCORING_METRIC, \
    NUMBER_OF_FOLDS


class DrugRepurposing:
    def __init__(self):
        """
        Drug repurposing for the SARS-CoV-2 pandemic -
        predicting whether an existing drug will be tested in clinical trials.
        """
        self.dr_covid_url = DR_COVID_RESULTS_URL
        self.dr_covid_filename = DR_COVID_RESULTS_FILENAME
        self.covid_clinical_trials_filename = COVID_CLINICAL_TRAILS_FILENAME
        self.drugs = list()
        self.dr_covid_results: Dict = dict()
        self.interventional_studies_df = None
        self.drug_to_clinical_trial_map = dict()
        self.data_set = dict()
        self.df = None

    def run(self):
        """
        main entry point
        build the data-set, print some debug information and run the model (grid search + model selected)
        """
        self._build_data_set()
        self._debug()
        self._run_model()

    def _build_data_set(self):
        """
        builds the data-set for the project
        data sources are Dr. covid results and clinical trails data source
        """
        self._get_dr_covid_results()
        self._save_drugs_list()
        self._load_clinical_trial_data()
        self._map_drug_to_clinical_trial()
        self._create_data_set()

    def _get_dr_covid_results(self):
        """
        load Dr. covid results
        """
        urllib.request.urlretrieve(url=self.dr_covid_url, filename=self.dr_covid_filename)
        with open(self.dr_covid_filename, "r") as f:
            self.dr_covid_results = json.load(f)

    def _save_drugs_list(self):
        """
        save valid drug list from Dr. covid results
        """
        drugs = set()
        for drug_name, score in self.dr_covid_results[0].items():
            if self.is_valid_drug_name(drug_name):
                drugs.add(drug_name)
        self.drugs = list(drugs)

    def _load_clinical_trial_data(self):
        """
        load clinical trial data source
        """
        covid_df = pd.read_csv(self.covid_clinical_trials_filename)
        self.interventional_studies_df = covid_df[covid_df[STUDY_TYPE_COLUMN] == STUDY_TYPE_INTERVENTIONAL]

    def _is_in_clinical_trial(self, name):
        """
        checks if a drug exists in clinical trails intervention
        """
        return any(self.interventional_studies_df[INTERVENTION_COLUMN].str.find(name.upper()) != -1)

    def is_valid_drug_name(self, drug_name):
        """
        checks if drug name is valid (by length)
        """
        return len(drug_name) < DRUG_NAME_MAX_LEN

    def _map_drug_to_clinical_trial(self):
        """
        add target label (is in clinical trails flag) to drug list
        """
        for drug in self.drugs:
            self.drug_to_clinical_trial_map[drug] = self._is_in_clinical_trial(name=drug)

    def _get_covid_to_drug_score(self, i, drug_name):
        """
        get disease-drug GNN score
        """
        return self.dr_covid_results[i][drug_name]

    def _create_data_set(self):
        """
        create main data-set
        """
        # Cols
        self.data_set[DRUG_NAME_COLUMN] = list()
        for i in range(len(self.dr_covid_results)):
            self.data_set[f'{COVID_VARIANT_PREFIX}_{i+1}'] = list()
        self.data_set[LABEL_COLUMN] = list()

        # Data
        for drug_name in self.drugs:
            self.data_set[DRUG_NAME_COLUMN].append(drug_name)
            for i in range(len(self.dr_covid_results)):
                self.data_set[f'{COVID_VARIANT_PREFIX}_{i+1}'].append(self._get_covid_to_drug_score(i, drug_name))
            self.data_set[LABEL_COLUMN].append(self.drug_to_clinical_trial_map[drug_name])

        try:
            self.df = pd.DataFrame.from_dict(self.data_set)
            self.df.to_csv(DRUG_REPURPOSING_DS_FILE_NAME)
        except Exception:
            raise DataCannotBeLoadedException()

    def _get_estimator(self, estimator_name: str):
        """
        get estimator (model) class by estimator (model) name
        """
        return ESTIMATORS[estimator_name]["estimator_object"]

    def _get_param_grid(self, estimator_name: str):
        """
        get grid search parameters by estimator (model) name
        """
        return ESTIMATORS[estimator_name]["param_grid"]

    def _grid_search(self, X, y):
        """
        run grid search and return best params.
        in order to test some other estimators, the value in SELECTED_ESTIMATOR_NAME should be changed in config.py
        """
        estimator = self._get_estimator(estimator_name=SELECTED_ESTIMATOR_NAME)
        param_grid = self._get_param_grid(estimator_name=SELECTED_ESTIMATOR_NAME)

        grid_search = GridSearchCV(estimator=estimator,
                                   param_grid=param_grid,
                                   scoring=SCORING_METRIC,
                                   cv=NUMBER_OF_FOLDS)
        grid_search.fit(X, y)

        print("best score = %3.2f" % (grid_search.best_score_))

        return grid_search.best_params_

    def _train_model(self, X, y, best_params):
        """
        train the model according to the best model selected in the grid search with the best parameters received
        """
        model = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', min_samples_split=10, splitter='random')

        model.fit(X, y)

        return model

    def _run_model(self):
        """
        run ML model
            load the data-set created
            split to train and test
            get best params by grid search
            train the model
            predict (train and test sets) and get score according to the metric selected
        """
        dataset_df = load_dataset(DRUG_REPURPOSING_DS_FILE_NAME)
        X = dataset_df.drop([LABEL_COLUMN, DRUG_NAME_COLUMN], axis=1)
        y = dataset_df[LABEL_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)

        best_params = self._grid_search(X_train, y_train)

        model = self._train_model(X_train, y_train, best_params)
        # Test set
        y_pred_test = model.predict(X_test)

        # Predictions (test set)
        test_score = roc_auc_score(y_test, y_pred_test)
        print("test score = %3.2f" % (test_score))

        # Predictions (train set)
        y_pred_train = model.predict(X_train)
        train_score = roc_auc_score(y_train, y_pred_train)
        print("train score = %3.2f" % (train_score))

    # some debug functions
    def _print_dr_covid_results_info(self):
        """
        debug Dr. covid data-source
        """
        print(f"Type of dr-covid results: {type(self.dr_covid_results)}")
        print(f"Length of dr-covid results: {len(self.dr_covid_results)}")
        print(f"Type of item in dr-covid results: {type(self.dr_covid_results[0])}")
        print(f"Length of item in dr-covid results: {len(self.dr_covid_results[0])}")

    def _print_drugs_list(self):
        """
        debug drug list
        """
        print(f"Drugs ({len(self.drugs)})")
        print(self.drugs[0])

    def _print_drug_to_clinical_trial(self):
        """
        debug drug to clinical trail map
        """
        print(list(self.drug_to_clinical_trial_map.items())[0])

    def _print_data_set_info(self):
        """
        debug the data-set
        """
        print(f"Data-set number of columns: {len(self.data_set)}")
        print(f"Data-set columns: {self.data_set.keys()}")

    def _debug(self):
        """
        print some debug information
        """
        self._print_dr_covid_results_info()
        self._print_drugs_list()
        self._print_drug_to_clinical_trial()
        self._print_data_set_info()


if __name__ == "__main__":
    print(datetime.datetime.now())
    DrugRepurposing().run()
    print(datetime.datetime.now())
