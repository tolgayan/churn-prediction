"""
Testing file for churn prediction
"""

import os
import logging
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with
    the other test functions
    '''
    try:
        data_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: \
                The file doesn't appear to have rows and columns")
        raise err

    return data_df


def test_eda(perform_eda, data_df):
    '''
    test perform eda function
    '''

    perform_eda(data_df)
    path = "./images/eda"

    try:
        assert os.path.exists(os.path.join(path, 'churn_distribution.png'))
        assert os.path.exists(os.path.join(
            path, 'customer_age_distribution.png'))
        assert os.path.exists(os.path.join(path, 'heatmap.png'))
        assert os.path.exists(os.path.join(
            path, 'marital_status_distribution.png'))
        assert os.path.exists(os.path.join(
            path, 'total_trans_ct_distribution.png'))
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: One of the eda plots could not \
                        generated successfully.")
        raise err


def test_add_churn_col(add_churn_col, data_df):
    '''
    test add churn col function
    '''
    data_df = add_churn_col(data_df)

    try:
        assert 'Churn' in data_df.columns
        logging.info("Testing add_churn_col: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing add_churn_col: Churn column is not added.")
        raise err

    return data_df


def test_encoder_helper(encoder_helper, data_df, category_lst):
    '''
    test encoder helper
    '''

    encoded_df = encoder_helper(data_df, category_lst)

    try:
        for category in category_lst:
            new_column_name = category + '_' + 'Churn'
            assert new_column_name in encoded_df.columns

        logging.info("Testing encoder_helper: SUCCESS")

    except AssertionError as err:
        logging.warning("Testing encoder_helper: \
                         Columns did not encoded successfully.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, data_df):
    '''
    test perform_feature_engineering
    '''

    x_train, x_test, y_train, y_test = perform_feature_engineering(data_df)

    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_feature_engineering: \
                        Dimensions of data and labels are wrong.")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''

    train_models(x_train, x_test, y_train, y_test)

    results_path = "./images/results"
    models_path = "./models"

    try:
        assert os.path.exists(os.path.join(
            results_path, 'feature_importances.png'))
        assert os.path.exists(os.path.join(
            results_path, 'logistic_results.png'))
        assert os.path.exists(os.path.join(results_path, 'rf_results.png'))
        assert os.path.exists(os.path.join(
            results_path, 'roc_curve_result.png'))
        logging.info("Testing train_models output plots: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing train_models output plots: \
                        One of the plots is not created")
        raise err

    try:
        assert os.path.exists(os.path.join(models_path, 'logistic_model.pkl'))
        assert os.path.exists(os.path.join(models_path, 'rfc_model.pkl'))
        logging.info("Testing train_models models: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing train_models models: \
                        One of the models is not saved")
        raise err


def main():
    """Run Tests"""
    data_df = test_import(cls.import_data)
    data_df = test_add_churn_col(cls.add_churn_col, data_df)

    columns_to_encode = ['Gender', 'Education_Level', 'Marital_Status',
                         'Income_Category', 'Card_Category']

    test_encoder_helper(cls.encoder_helper, data_df, columns_to_encode)

    test_eda(cls.perform_eda, data_df)
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, data_df)
    test_train_models(cls.train_models, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
