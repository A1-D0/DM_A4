'''
Description: 
How to run: python assignment_4.py <filepath> <test_size> <save_pipeline>
Authors: Bradyen Miller, Osvaldo Hernandez-Segura
References: ChatGPT, Numpy documentation, Pandas documentation, Scikit-Learn documentation, joblib documentation
'''
import pandas as pd
# import numpy as np
import os
import sys
import utils

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from transformers import CustomBestFeaturesTransformer, CustomDropNaNColumnsTransformer, CustomClipTransformer, CustomReplaceInfNanWithZeroTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import Memory

def get_pipeline(X: pd.DataFrame)-> Pipeline:
    '''
    Get the custom pipeline.

    :param X: the input features.
    :return: the pipeline.
    '''
    mem = Memory(location='cache_dir', verbose=0)
    pipeline = Pipeline(
        steps=[ 
                ("drop_nan", CustomDropNaNColumnsTransformer(threshold=0.6)),
                ("inf_nan", CustomReplaceInfNanWithZeroTransformer()),
                ("clipper", CustomClipTransformer()),
                ("scaler", StandardScaler()),
                # ("rfecv", CustomBestFeaturesTransformer()), # comment this out for testing only; must be in final pipeline
                ("classifier", None)
                ],
                memory=mem # cache transformers (to avoid fitting transformers multiple times)
                )
    return pipeline

def run_pipeline(X: pd.DataFrame, y: pd.DataFrame, classifiers: list)-> Pipeline:
    '''
    Run the pipeline.

    :param data: the data to process.
    :return: the pipeline.
    '''
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
        model_name = pipeline.named_steps['classifier'].__class__.__name__
        text = "%s: Accuracy: %.3f\n" % (model_name, accuracy_score(y_test, predictions)) 
        text += "%s: AUC: %.3f" % (model_name, roc_auc_score(y_test, predictions)) 
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

    utils.data_understanding(data, "data_quality_report.txt", save=False)

    X, y = utils.get_X_y(data=data, drop_X_columns=['CID', 'Name', 'Inhibition'], target='Inhibition')
    classifiers = [DummyClassifier(strategy='most_frequent'), 
                   SGDClassifier(loss='log_loss'),
                   GaussianNB(),
                   SVC(probability=True),
                   RandomForestClassifier(n_jobs=-1),
                   LogisticRegression()
                ]
    pipeline = run_pipeline(X, y, classifiers)
    if int(argv[2]) == 1: utils.save_pipeline_to_dump(pipeline, os.path.join(os.pardir, "output"), "modeling_pipeline")


    # predict on test data using pipeline (Cf. test_pipeline_predictions.py)


    # get top 100 ranking and bottom 100 ranking (Cf. test_pipeline_predictions.py)


    exit(0)

if __name__ == '__main__':
    main()