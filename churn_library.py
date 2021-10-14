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
import pandas as pd
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
    plt.clf()


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


def classification_report_image(lrc,
                                cv_rfc,
                                y_train,
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
    generate_report('Random Forest', 
                    y_train, y_test,
                    y_train_preds_rf, y_test_preds_rf,
                    output_path='images/results/rf_results.png')

    generate_report('Logistic Regression', 
                    y_train, y_test,
                    y_train_preds_lr, y_test_preds_lr,
                    output_path='images/results/logistic_results.png')

    generate_roc_curve(lrc, cv_rfc, y_test,
                       output_path='images/results/roc_curve_result.png')


def generate_report(experiment_name,
                    y_train_gt,
                    y_test_gt,
                    y_train_preds,
                    y_test_preds,
                    output_path):
    """
    Produces plot for given experiment name and predictions

    input:
        experiment_name: name of the experiment, 
            such as Logistic Regression or Random Forest
        y_train_gt: ground truth labels for train set 
        y_test_gt: ground truth labels for tes set
        y_train_preds: predictions for train set
        y_test_preds: predictions for test set
        output_path: path to save the result plot

    """

    train_name = '%s Train' % experiment_name
    test_name = '%s Test' % experiment_name

    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, train_name,
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test_gt, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, test_name, {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train_gt, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.clf()


def generate_roc_curve(lrc, cv_rfc, y_test,
                       output_path='images/roc_curve_result.png'):
    """
    Plots ROC curve for logistic regression and random forest results,
    and saves to the given path.

    input:
        lrc: logistic regression model
        cv_rfc: random forest model
        y_tess: test data labels
        output_path: path to save the result plot
    """
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(output_path)
    plt.clf()


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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    fig = shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(output_pth, bbox_inches='tight')
    plt.clf()



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
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    classification_report_image(lrc,
                                cv_rfc,
                                y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc.best_estimator_, 
                            X_test, 
                            'images/results/feature_importances.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    df = import_data('data/bank_data.csv')
    df = add_churn_col(df)
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)
