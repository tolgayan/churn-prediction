"""

TODO:

author: E. Tolga Ayan
Date: 29 Sep 2021

"""

# import libraries
import shap
import joblib
import os
from collections import namedtuple
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
                    pth: a path to the csv
    output:
                    df: pandas dataframe
    '''

    return pd.read_csv(pth)


def add_churn_col(df):
    """
    Compute and add Churn data to the given dataframe.

    input:
                    df: pandas dataframe
    """

    def churn_function(val): return 0 if val == "Existing Customer" else 1
    df['Churn'] = df['Attrition_Flag'].apply(churn_function)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
                    df: pandas dataframe

    output:
                    None
    '''

    Column = namedtuple("Column", ['col_name', 'is_categorical'])

    columns = []
    cat_col_names = ['Churn', 'Marital_Status']
    quant_col_names = ['Customer_Age', 'Total_Trans_Ct']

    columns.extend([Column(col_name, True) for col_name in cat_col_names])
    columns.extend([Column(col_name, False) for col_name in quant_col_names])

    for column in columns:
        column_name = column.col_name
        is_categorical = column.is_categorical
        filename = f'{column_name.lower()}_distribution.png'
        output_path = os.path.join(*['images', 'eda', filename])

        _plot_and_save(df, column_name, is_categorical, output_path)

    # Generate heatmap and save
    output_path = os.path.join(*['images', 'eda', 'heatmap.png'])
    _heatmap(df, output_path)


def _plot_and_save(df, column_name, is_categorical, output_path):
    '''
    Generate a countplot, or histplot for a column of a given dataframe, and
    save the result.
    input:
                    df: pandas dataframe
                    column_name: name of the column to plot
                    is_categorical: If True, generate countplot. 
                            Else, generate histplot.
                    output_path: Path to save the result plot. 

    output:
                    None
    '''

    plt.figure(figsize=(20, 10))
    if is_categorical:
        plot = sns.countplot(x=column_name, data=df)
    else:
        plot = sns.histplot(x=column_name, data=df, kde=True)

    plot.figure.savefig(output_path)
    plt.clf()


def _heatmap(df, output_path):
    """
    Generate and save heatmap for a given pandas dataframe
    input:
                    df: pandas dataframe
                    output_path: path to save the result plot.

    output:
                    None
    """

    plt.figure(figsize=(20, 10))
    plot = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plot.figure.savefig(output_path)


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with 
    propotion of churn for each category - associated with cell 15 from the 
    notebook.

    input:
                    df: pandas dataframe
                    category_lst: list of columns that contain categorical features
                    response: string of response name [optional argument that could be 
                            used for naming variables or index y column]

    output:
                    df: pandas dataframe with new columns for
    '''

    for category in category_lst:
        propotion_vals = []
        groups = df.groupby(category).mean()['Churn']

        for val in df[category]:
            propotion_vals.append(groups.loc[val])

        new_column_name = category + '_' + response
        df[new_column_name] = propotion_vals

    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
                      df: pandas dataframe
                      response: string of response name [optional argument that could 
                            be used for naming variables or index y column]

    output:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    '''

    columns_to_encode = ['Gender', 'Education_Level', 'Marital_Status',
                         'Income_Category', 'Card_Category']

    df = encoder_helper(df, columns_to_encode, response)

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols] 
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores 
            report as image in images folder
    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

    output:
                     None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
                    model: model object containing feature_importances_
                    X_data: pandas dataframe of X values
                    output_pth: path to store the figure

    output:
                     None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    output:
                      None
    '''
    pass


if __name__ == '__main__':
    df = import_data('data/bank_data.csv')
    df = add_churn_col(df)
    perform_eda(df)
