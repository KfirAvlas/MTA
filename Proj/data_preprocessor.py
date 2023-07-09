import numpy as np
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Proj.config import NUM_OF_COVID_VARIANTS, COVID_VARIANT_PREFIX, DRUG_NAME_COLUMN


class DataPreprocessor(object):
    def __init__(self, estimator=None):
        self.estimator = estimator

    # def _create_pipeline(self, estimator):
    #     numerical_columns = list()
    #
    #     for i in range(NUM_OF_COVID_VARIANTS):
    #         numerical_columns.append(f'{COVID_VARIANT_PREFIX}_{i+1}')
    #
    #     numerical__pipeline = Pipeline([
    #         ('std_scaler', StandardScaler())
    #     ])
    #
    #     column_transformer = ColumnTransformer(
    #         transformers=[
    #             ("drop_drug_name_col", 'drop', DRUG_NAME_COLUMN),
    #             ("num_pip", numerical__pipeline, numerical_columns)
    #         ]
    #     )
    #
    #     steps = [ ("column_transformer", column_transformer)]
    #     if estimator:
    #         steps.append(("estimator", estimator))
    #
    #     return Pipeline(steps=steps)

    def create_pipeline(self):
        scaler = StandardScaler()
        steps = [("scaler", scaler)]
        if self.estimator:
            steps.append(("estimator", self.estimator))

        self.pipe = Pipeline(steps)

    def fit(self, df: DataFrame):
        self.pipe.fit(df)

    def transform(self, df):
        return self.pipe.transform(df)
