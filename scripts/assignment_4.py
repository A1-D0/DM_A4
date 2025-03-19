'''
Description: 
How to run: python assignment_4.py <filepath> <test_size> <save_pipeline>
Authors: Bradyen Miller, Osvaldo Hernandez-Segura
References: ChatGPT, Numpy documentation, Pandas documentation
'''
import pandas as pd
import numpy as np
import os
import sys
import utils

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from transformers import CustomBestFeaturesTransformer, CustomImputerTransformer, CustomDropNaNColumnsTransformer, CustomClipTransformer, CustomReplaceInfNanWithZeroTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

def get_X_y(data: pd.DataFrame, drop_X_columns: list, target: str | int)-> tuple:
    '''
    Get the features and target.

    :param data: the data.
    :param drop_X_columns: the columns to drop from the input features.
    :param target: the target column.
    :return: X and y.
    '''
    X = data.drop(columns=drop_X_columns)
    if isinstance(target, int): y = data.iloc[:, target]
    else: y = data[target]
    return X, y

def get_pipeline(X: pd.DataFrame)-> Pipeline:
    '''
    Get the custom pipeline.

    :param X: the input features.
    :return: the pipeline.
    '''
    numerical_transformer = Pipeline(
        steps=[
                ("clipper_1", CustomClipTransformer()),
                ("scaler", StandardScaler()),
                # ("imputer", CustomImputerTransformer()),
                # ("clipper_2", CustomClipTransformer())
                ])

    preprocessor = ColumnTransformer(
        transformers=[  
                        ("inf_nan", CustomReplaceInfNanWithZeroTransformer(), X.columns),
                        # ("num", numerical_transformer, X.columns),
                        ("clipper", CustomClipTransformer(), X.columns),
                        ("scaler", StandardScaler(), X.columns)
                        # ("rfecv", CustomBestFeaturesTransformer(), X.columns)
                        ])

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),
                ("classifier", None)])
    return pipeline

def run_pipeline(data: pd.DataFrame)-> Pipeline:
    '''
    Run the pipeline.

    :param data: the data to process.
    :return: the pipeline.
    '''
    X, y = get_X_y(data=data, drop_X_columns=['CID', 'Name', 'Inhibition'], target='Inhibition')

    # for testing pipeline preprocessing steps
    X = np.clip(X, a_min=-1e9, a_max=1e9)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    classifiers = [DummyClassifier(strategy='most_frequent'), SGDClassifier(), RandomForestClassifier(n_jobs=-1)]

    pipeline = get_pipeline(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    output_path = os.path.join(os.pardir, "output")
    os.makedirs(output_path, exist_ok=True)
    write_to_path = os.path.join(output_path, "results.txt")
    write_to_file = open(write_to_path, "w", encoding="ascii")

    for idx in range(len(classifiers)):
        pipeline.set_params(classifier=classifiers[idx])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        text = "%s: Accuracy: %.3f" % (pipeline.named_steps['classifier'].__class__.__name__, accuracy_score(y_test, predictions)) 
        print(text)
        write_to_file.write(text + "\n")

    write_to_file.close()
    print(f"Results saved to {write_to_path}")
    return pipeline

def main()-> None:
    argv = sys.argv[1:]
    if len(argv) != 3: # update for unseen data prediction section
        print("Enter the filepath of the dataset, the test size (0 for all data), and whether to save pipeline (0 or 1)")
        exit(1)
    elif not os.path.exists(argv[0]):
        print("Invalid filepath of the train dataset")
        exit(1)
    elif argv[2] not in ['0', '1']:
        print("Invalid save pipeline option")
        exit(1)
    
    test_size = int(argv[1])
    data = pd.read_table(argv[0], low_memory=False)
    if test_size > 0 and test_size < len(data): 
        print("Testing...")
        data = data.iloc[:test_size]

    # utils.data_understanding(data, "data_quality_report.txt", save=False)

    pipeline = run_pipeline(data)
    if int(argv[2]) == 1: utils.save_pipeline_to_dump(pipeline, os.path.join(os.pardir, "output"), "modeling_pipeline")


    # predict on test data using pipeline
    # unseen_data = pd.read_csv(argv[2], low_memory=False)


    # get top 100 ranking and bottom 100 ranking


    exit(0)

if __name__ == '__main__':
    main()