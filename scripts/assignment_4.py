'''
Description: 
Authors: Bradyen Miller, Osvaldo Hernandez-Segura
References: ChatGPT, Numpy documentation, Pandas documentation
'''
import pandas as pd
import numpy as np
import os
import sys
import joblib
import utils

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def save_pipeline_to_dump(pipeline: Pipeline, output_path)-> None:
    '''
    Save the pipeline to a dump file.

    :param pipeline: the pipeline to save.
    :param output_path: the path to save the pipeline.
    :return: None.
    '''
    dump_to_path = os.path.join(output_path,"pipeline.pkl")
    joblib.dump(pipeline, dump_to_path)
    print(f"Pipeline saved to {dump_to_path}")
    return None

def run_pipeline(data: pd.DataFrame)-> Pipeline:
    '''
    Run the pipeline.

    :param data: the data to process.
    :return: the pipeline.
    '''

    model_names = ['DummyClassifier', 'NB']

    






    return None


def main()-> None:
    argv = sys.argv[1:]
    if len(argv) != 3:
        print("Enter the filepath of the dataset, the test size (0 for all data), and whether to save pipeline (0 or 1)")
        exit(1)
    elif not os.path.exists(argv[0]):
        print("Invalid filepath of the train dataset")
        exit(1)
    elif argv[2] not in ['0', '1']:
        print("Invalid save pipeline option")
        exit(1)
    
    test_size = int(argv[1])
    data = pd.read_csv(argv[0])
    if test_size > 0: 
        print("Testing...")
        data = data.iloc[:test_size]

    utils.data_understanding(data, "data_quality_report.txt", save=False)


    


    pipeline = run_pipeline(data)

    if argv[2] == 1: save_pipeline_to_dump(pipeline, os.path.join(os.pardir, "output"))

    exit(0)

if __name__ == '__main__':
    main()