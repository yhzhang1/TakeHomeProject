import os
import pandas as pd
project_dir = os.path.dirname(__file__)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
import utils
import matplotlib.pyplot as plt

def main():
    df = utils.load_data()
    X = df.drop(columns=['weight','label'])
    y = df.label
    sample_weight = df.weight
    le = LabelEncoder()
    y = le.fit_transform(y)

    train_data, validation_data, test_data = split_dataset(X, y, sample_weight)
    preprocessed_X_train, preprocessed_X_validation, preprocessed_X_test, feature_names = preprocess(train_data['X'], validation_data['X'], test_data['X'])

    logistic_regression = LogisticRegression()
    random_forest = RandomForestClassifier(verbose=2, n_jobs=-1)
    xgb = XGBClassifier(early_stopping_rounds=100, n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')

    models = {
        'logistic_regression':logistic_regression,
        'random_forest':random_forest,
        'xgb':xgb
        }

    for model_name in models:
        print(f'beging training {model_name}...')
        if model_name == 'xgb':
            train_xgb(models[model_name], preprocessed_X_train, train_data['y'], train_data['sample_weight'], preprocessed_X_validation, validation_data['y'])
        else:
            train(models[model_name], preprocessed_X_train, train_data['y'], train_data['sample_weight'])
        print(f'finished training {model_name}')

    for model_name in models:
        print('-'*100)
        print(f'{model_name} test result')
        y_pred = models[model_name].predict(preprocessed_X_test)
        y_prob = models[model_name].predict_proba(preprocessed_X_test)[:,1]
        report(y_pred, y_prob, test_data, le)

    feature_importance(models['xgb'], feature_names)

def split_dataset(X, y, sample_weight):
    X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test = train_test_split(X, y, sample_weight, test_size=0.3, stratify=y)
    X_validation, X_test, y_validation, y_test, sample_weight_validation, sample_weight_test = train_test_split(X_test, y_test, sample_weight_test, test_size=2/3, stratify=y_test)

    train = {'X':X_train, 'y':y_train, 'sample_weight':sample_weight_train}
    validation = {'X':X_validation, 'y':y_validation, 'sample_weight':sample_weight_validation}
    test = {'X':X_test, 'y':y_test, 'sample_weight':sample_weight_test}

    return train, validation, test

def preprocess(X_train, X_validation, X_test):
    num_cols = X_train.select_dtypes(exclude='object').columns
    cat_cols = X_train.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])

    preprocessor.fit(X_train)
    preprocessed_X_train = preprocessor.transform(X_train)
    preprocessed_X_validation = preprocessor.transform(X_validation)
    preprocessed_X_test = preprocessor.transform(X_test)

    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    feature_names = np.concatenate([num_cols, cat_feature_names])

    return preprocessed_X_train, preprocessed_X_validation, preprocessed_X_test, feature_names

def train(model, preprocessed_X_train, y_train, sample_weight_train):
    model.fit(preprocessed_X_train, y_train, sample_weight=sample_weight_train)

def train_xgb(model, preprocessed_X_train, y_train, sample_weight_train, preprocessed_X_validation, y_validation):
    model.fit(preprocessed_X_train, y_train, sample_weight=sample_weight_train, eval_set=[(preprocessed_X_validation, y_validation)], verbose=True)

def feature_importance(xgb, feature_names):
    importances = xgb.feature_importances_
    indices  = np.argsort(importances)[::-1][:5]

    print('-'*100)
    print('feature importance:\n')
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.3f}")

    sorted_importance = pd.Series([importances[i] for i in indices], index=[feature_names[i] for i in indices])
    plt.figure(figsize=(10, 6))
    sorted_importance.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Top 5 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def report(y_pred, y_prob, test_data, labelencoder):
    print(classification_report(y_true=test_data['y'], y_pred=y_pred, sample_weight=test_data['sample_weight'], target_names=labelencoder.classes_))
    print("ROC AUC:", roc_auc_score(test_data['y'], y_prob, sample_weight=test_data['sample_weight']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true=test_data['y'], y_pred=y_pred, sample_weight=test_data['sample_weight']))

if __name__ == '__main__':
    main()