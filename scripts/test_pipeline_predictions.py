'''
Description: This program runs the pipeline predictions.
Authors: Bradyen Miller, Osvaldo Hernandez-Segura
References: ChatGPT, Numpy documentation, Pandas documentation, Scikit-Learn documentation, joblib documentation
'''
import pandas as pd
import numpy as np
import os
import joblib

def main()-> None:

    pipeline_path = os.path.join(os.pardir, 'output', 'modeling_pipeline.pkl')
    pipeline = None
    try:
        with open(pipeline_path, 'rb') as f:
            pipeline = joblib.load(f)
    except:
        print("Pipeline not found.")
        exit(1)
    
    data = pd.read_csv(os.path.join(os.pardir, 'data', 'new_molecules.csv'), low_memory=False)

    print("Before conversion:")
    print(data.info())
    print(data.select_dtypes(include='object').columns)
    print(data.select_dtypes(include='object').head(3))

    print("After conversion:")
    data = data.apply(pd.to_numeric, errors='coerce')
    print(data.info())
    print(data.select_dtypes(include='object').columns)
    print(data.select_dtypes(include='object').head(3))

    X = data.drop(columns=['id']) # dataset must start with nAcid, ALogP, ALogp2

    # predict on test data using pipeline
    y_pred = pipeline.predict(X)
    y_score = None
    results = pd.DataFrame(data={'id': data['id'], 'prediction': y_pred, 'score': y_score})

    try:
        y_score = pipeline.predict_proba(X)[:, 1] # select class 1 -- inhibition
        results['score'] = y_score
        results.sort_values(by='score', ascending=False, inplace=True) # get top 100 ranking and bottom 100 ranking via sorting
    except:
        print("Pipeline does not support predict_proba.")
    
    output_path = os.path.join(os.pardir, 'output', 'ranked_predictions_test.csv')
    results.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

    exit(0)

if __name__ == "__main__":
    main()