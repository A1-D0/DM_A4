'''
Description: 
Authors: Bradyen Miller, Osvaldo Hernandez-Segura
References: ChatGPT, Numpy documentation, Pandas documentation
'''
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.pipeline import Pipeline

def save_pipeline_to_dump(pipeline: Pipeline, output_path: str, file_name: str="pipeline")-> None:
    '''
    Save the pipeline to a dump file as a pkl file.

    :param pipeline: the pipeline to save.
    :param output_path: the path to save the pipeline.
    :param file_name: the name of the file.
    :return: None.
    '''
    dump_to_path = os.path.join(output_path, f"{file_name}.pkl")
    joblib.dump(pipeline, dump_to_path)
    print(f"Pipeline saved to {dump_to_path}")
    return None

def write_to_file(text: str, output_path: str)-> None:
    '''
    Write text to file.

    :param text: str
    :param output_path: str
    :return: None
    '''
    path = os.path.join(os.pardir, 'output')
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, output_path)
    with open(output_path, 'w') as f:
        f.write(text)
    print(f"Text saved to {output_path}")

def data_understanding(data: pd.DataFrame, output_path: str, save: bool=False)-> None:
    '''
    Data understanding.

    :param data: pd.DataFrame
    :param output_path: name to use for the saved output file
    :param save: whether to save the output
    :return: None
    '''
    dataset_name = output_path.split('_data_desc')[0]
    text = f"{dataset_name} data description:\n" + f"(Rows, Columns): {data.shape}\n"
    text += f"Duplicated rows: {data.duplicated().sum()}\n" + f"Number of rows with missing values: {data.isnull().any(axis=1).sum()}\n"
    text += f"Number of columns with missing values: {data.isnull().any(axis=0).sum()}\n\n" + f"{data.dtypes.to_string()}\n\n" + f"{data.describe().T.to_string()}\n"
    print(text)
    if save: write_to_file(text, output_path)

def get_X_y(data: pd.DataFrame, drop_X_columns: list, target: str | int)-> tuple:
    '''
    Get the features and target. Returns X and y if the columns parameters are valid.

    :param data: the data.
    :param drop_X_columns: the columns to drop from the input features.
    :param target: the target column.
    :return: X and y.
    '''
    X, y = None, None
    try:
        X = data.drop(columns=drop_X_columns)
    except:
        print("Invalid columns to drop.")
    if isinstance(data, pd.DataFrame):
        if isinstance(target, int): y = data.iloc[:, target]
        else: y = data[target]
    else:
        if isinstance(target, int): y = data[:, target]
        else: print("Invalid target column for %s" % type(data))
    return X, y