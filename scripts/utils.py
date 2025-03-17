'''
Description: 
Author: Bradyen Miller, Osvaldo Hernandez-Segura
References: ChatGPT, Numpy documentation, Pandas documentation
'''
import pandas as pd
import numpy as np
import os

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