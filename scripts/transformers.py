'''
Description: This program contains three CustomTransformer classes. CustomTransformer is used to remove features that have a high correlation with the target feature and features with zero variance.
CustomInterpolateTransformer.
CustomClipTransformer.
Author: Osvaldo Hernandez-Segura
References: ChatGPT, Scikit-Learn, Numpy documentation, GeekforGeeks
'''
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class CustomTransformer(BaseEstimator, TransformerMixin): # use check_quality data leakage function for inspiration
    def __init__(self, corr_threshold=0.75): 
        self.corr_threshold = corr_threshold
        self.leaked_columns_ = np.array([]) # store columns leaking information (note: this variable is named as such because of scikit learn naming conventions for transformers Cf. fit warning)

    def find_high_correlation_features(self, X, y):
        '''
        Find features with high correlation to the target feature.
        '''
        data = np.concatenate((X, y), axis=1)
        correlation_matrix = np.corrcoef(data, rowvar=False)
        correlation_vector = np.absolute(correlation_matrix[:-1, -1])
        largest_coef_features = np.where(correlation_vector >= self.corr_threshold)[0]
        return largest_coef_features

    def fit(self, X, y=None):
        '''
        Identify columns leaking the information.
        '''
        # print("Fitting CustomTransformer...")
        if y is None: return self
        X, y = np.asarray(X), np.asarray(y).reshape(-1, 1)
        self.leaked_columns_ = self.find_high_correlation_features(X, y)
        return self 

    def transform(self, X):
        '''
        Remove columns leaking the information.
        '''
        # print("Transforming with CustomTransformer...")
        if len(self.leaked_columns_) > 0: X = np.delete(X, self.leaked_columns_, axis=1) # drop leaked features
        return X

class CustomBestFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cv: int=5, scoring: str='accuracy'): 
        self.rfecv_ = None
        self.cv = cv
        self.scoring = scoring
        # self.best_features = []

    def fit(self, X, y=None):
        '''
        Fit the RFECV.
        '''
        if y is None: return self
        print("Fitting RFECV...")
        self.rfecv_ = RFECV(estimator=RandomForestClassifier(n_jobs=-1), step=1, cv=self.cv, scoring=self.scoring)
        self.rfecv_.fit(X, y)
        # self.best_features = X.columns[self.rfecv.support_]
        return self 

    def transform(self, X):
        '''
        Transform the data using trained RFECV.
        '''
        print("Transforming data using RFECV...")
        try:
            return X[:, self.rfecv_.support_]
        except:
            print("X data dimensions not compatible with RFECV.")
            return X

class CustomImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: int=-1e9, upper: int=1e9):
        self.lower = lower
        self.upper = upper 
        self.imp_ = None

    def fit(self, X, y=None):
        '''
        Fit the IterativeImputer.
        '''
        print("Fitting IterativeImputer...")
        self.imp_ = IterativeImputer(random_state=0, min_value=self.lower, max_value=self.upper)
        self.imp_.fit(X) if y is None else self.imp_.fit(X, y)
        return self 

    def transform(self, X):
        '''
        Transform the data using trained IterativeImputer.
        '''
        print("Transforming data using IterativeImputer...")
        return self.imp_.transform(X)
    
class CustomClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: int=-1e9, upper: int=1e9): 
        self.lower = lower
        self.upper = upper
        pass

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        '''
        Clip large values.
        '''
        X = np.clip(X, a_min=self.lower, a_max=self.upper)
        return X

class CustomDropNaNColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: int=0.5): 
        self.nan_columns = []
        self.threshold = threshold

    def fit(self, X, y=None):
        '''
        Find columns with NaN values.
        '''
        if isinstance(X, pd.DataFrame): self.nan_columns = (X.isnull().sum() > self.threshold*len(X))
        else: self.nan_columns = np.array(X[X.isnull().sum() > self.threshold*len(X)]).reshape(1, -1)
        return self

    def transform(self, X):
        '''
        Drop columns with NaN values.
        '''
        if len(self.nan_columns) > 0: 
            if isinstance(X, pd.DataFrame): X = X.loc[:, ~self.nan_columns]
            else: X = X[:, ~self.nan_columns]
        return X

class CustomReplaceInfNanWithZeroTransformer(BaseEstimator, TransformerMixin):
    def __init__(self): 
        pass

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        '''
        Clip large values.
        '''
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        return X