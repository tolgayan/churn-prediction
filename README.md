# Customer Churn Prediction

## About <a name = "about"></a>

Udacity Machine Learning Devops Engineer Course Project 1: Customer Churn Prediction.

The purpose of this project is to apply code formatting and testing best practises in a data science project. In the project, customer churns are predicted using random forest and logistic regression models.


## Getting Started <a name = "getting_started"></a>

The file `churn_notebook.ipynb` is provided by Udacity.

Project structure is as follows:

```
churn-prediction/
    data/
        bank_data.csv
    images/
        eda/     # eda plots here
        results/ # training result plots here
    logs/
    models/
    churn_library.py
    churn_notebook.ipynb
    churn_script_logging_and_tests.py
    LISENCE
    README.md
    requirements.txt
```

### Prerequisites

For installation of the packages, run `pip install -r requirements.txt`


## Usage <a name = "usage"></a>

The preprocessing and training pipeline can be used either by running `churn_script_logging_and_tests.py` or `churn_library.py`. The difference between them is the first one also run tests, and reports to a log file.

