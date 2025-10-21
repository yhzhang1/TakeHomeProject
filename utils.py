import os
project_dir = os.path.dirname(__file__)
import pandas as pd

def load_data():
    with open(f'{project_dir}/census-bureau.columns') as f:
        columns = [line.strip() for line in f if line.strip()]

    df = pd.read_csv(f'{project_dir}/census-bureau.data', names=columns)
    df = clean_data(df)

    return df

def clean_data(df):
    has_na = df.isna().any().any()
    print('has_na: ', has_na)
    na_rows = df[df.isna().any(axis=1)]
    has_na_cols = df.columns[df.isna().any()]
    print('has_na_cols: ', has_na_cols)
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)
    cat_cols = df.select_dtypes(exclude=['number']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    assert not df.isna().any().any()

    return df