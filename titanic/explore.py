# Import libraries
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import svm
import seaborn as sn
import pandas as pd
import numpy as np


def list_missing_features_fraction(df):
    '''
    Prints the list of missing values in a given dataframe by descended sorting

    Parameters
    ----------
    df : Pandas DataFrame object
        The given dataset.

    Returns
    -------
    None.

    '''
    # Get the percentage of missing values for each column
    missing_frac = 1 - df.count() / len(df)
    return missing_frac[missing_frac > 0.0].sort_values(ascending=False)


def get_columns_names_has_missing(df):
    '''
    Returns the list of columns where the column has missing values

    Parameters
    ----------
    df : Pandas DataFrame object
        The given dataset.

    Returns
    -------
    list
        The list of columns where the column has missing values.

    '''
    return df.columns[df.isnull().any()].tolist()


def gen_info(df):
    ''' Prints generic information about the given frame'''

    # Describe frame
    print(df.describe())
    print('\n')

    # Print column names
    print(df.columns)
    print('\n')

    # Print dtypes
    print(df.dtypes)
    print('\n')

    # Print column names which have missing values
    print('Columns which have missing values')
    print(get_columns_names_has_missing(df))
    print('\n')

    # Print frame length
    print('Length of the frame:', len(df))
    print('\n')

    # Print missing feature fraction
    print('Percentage of the missing values regarding to related columns')
    print(list_missing_features_fraction(df))


def draw_corr_matrix(df):
    '''
    Draws a correlation matrix by using the giving frame

    Parameters
    ----------
    df : Pandas Frame
        Frame to be visualized its correlation.

    Returns
    -------
    None.

    '''
    # Plot correlation matrix
    sn.heatmap(df.corr(), annot=True)
    # Show plot
    plt.show()


def anova(df, f: str, test='F'):
    '''
    Runs the ANOVA test.

    Parameters
    ----------
    df : object
        Pandas Frame.
    f : str
        Formula.
    test : string
        Type of test in ANOVA.

    Returns
    -------
    table : object
        ANOVA result table.

    '''
    # Create linear model
    lm = ols(f, data=df).fit()
    # Run ANOVA
    table = sm.stats.anova_lm(lm, test=test, typ=2, robust='hc3')
    # Return table
    return table


def chi_test_with_all_categorical_features(df):

    # Extract categorical features from frame
    cat_columns = df.select_dtypes(include='category').columns
    # Create an empty list for results
    chi2_check = list()
    # Check each categorical column with target feature
    for col in cat_columns:
        # If p value is less than 5%
        if chi2_contingency(pd.crosstab(df['Survived'], df[col]))[1] < 0.05:
            # Reject the null hypothesis
            chi2_check.append('Reject Null Hypothesis')
        # If not
        else:
            # Fail to reject
            chi2_check.append('Fail to Reject Null Hypothesis')
    # Set results
    results = pd.DataFrame(data=[cat_columns, chi2_check]).T
    results.columns = ['Column', 'Hypothesis']
    # Return results
    return results


def convert_categorical2Binary(df, columns: list):
    '''
    Creates dummy variables as binary to the given column list

    Parameters
    ----------
    df: object
        Pandas Frame.
    columns: list
        List of columns that will be converted

    Returns
    -------
    df: Pandas Frame
        Original frame that is concatenated with the dummy variables.
    '''
    dummies = pd.get_dummies(df, columns=columns)
    # Return frame
    return pd.concat([df.drop(columns, axis=1, inplace=True), dummies], axis=1)


def split_X_y(df, feature: str):
    '''
    Split given frame into X, y by considering missing/non-missing

    Parameters
    ----------
    df: object
        Pandas Frame.
    feature: str
        Feature that is going to be y.

    Returns
    -------
    X_train: object
        Pandas Frame.
    y_train: object
        Pandas Frame.
    X_test: object
        Pandas Frame.

    '''
    # Set X_test as missing rows
    X_test = df[df[feature].isnull()]
    # Get all columns except target
    X_test = X_test.loc[:, df.columns != feature]

    # Set the difference for X_train and X_test
    diff = df.index.difference(X_test.index)
    # Set X_train
    X_train = df.loc[diff, df.columns != feature]
    # Set y_train
    y_train = df.loc[diff, feature]

    # Return X, y
    return X_train, y_train, X_test


def classification_with_SVM(X_train, y_train, X_test):
    '''
    Prediction with SVM

    Parameters
    ----------
    X_train: object
        Pandas Frame.
    y_train: object
        Pandas Series.
    X_test: object
        Pandas Frame.

    Returns
    -------
    predictions: ndarray of shape
        Returns the log-probabilities of the sample for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes_
    '''
    # Apply SVM
    model_SVM = svm.SVC()
    # Fit the model
    model_SVM.fit(X_train, y_train)

    # Predict the test samples
    predictions = model_SVM.predict(X_test)

    # Return predictions
    return predictions


def regression_with_KNN(X_train, y_train, X_test):
    '''
    Prediction with SVM

    Parameters
    ----------
    X_train: object
        Pandas Frame.
    y_train: object
        Pandas Series.
    X_test: object
        Pandas Frame.

    '''
    # Create the model
    knn_reg = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    # Fit the model
    knn_reg.fit(X_train, y_train)

    # Predict
    predictions = knn_reg.predict(X_test)

    # Return predictions
    return predictions


def process(df):
    '''
    Process giving frame for further analytical operations

    Parameters
    ----------
    df : object
        Pandas Frame.

    Returns
    -------
    object
        Pandas Frame.

    '''
    # Change types of features
    # Fill NaN values as np.nan not to get error
    df.fillna(np.nan, inplace=True)
    # Set new types for certain features
    types = {
        'PassengerId': 'category',
        'Survived': 'category',
        'Pclass': 'category',
        'Sex': 'category',
        'SibSp': 'category',
        'Parch': 'category',
        'Ticket': 'category',
        'Cabin': 'category',
        'Embarked': 'category'
    }
    # Apply new types
    df = df.astype(types)

    # Drop 'Name' feature
    df2 = df.drop(['Name'], axis=1)

    # Return df2
    return df2


def fill_embarked(df):
    '''
    Whole process of the prediction of Embarked feature

    Parameters
    ----------
    df: object
        Pandas Frame

    '''
    # Prepare dataset for predicting Embarked
    train_emb = df.drop(['PassengerId', 'Cabin'], axis=1)
    # Drop NaN (from Age column) rows
    train_emb.dropna(subset=['Age'], inplace=True)
    # Prepare column list for converting categorical data to binaries
    categorical = train_emb.select_dtypes(include='category').columns.drop('Embarked')

    # Create dummy variables for SVM prediction
    train_emb = convert_categorical2Binary(train_emb, categorical)

    # Split data as X, y
    X_train_emb, y_train_emb, X_test_emb = split_X_y(train_emb, 'Embarked')

    # Predict the missing Embarked values
    predictions = classification_with_SVM(X_train_emb, y_train_emb, X_test_emb)

    # Add predictions to original frame
    df.loc[df.Embarked.isnull(), 'Embarked'] = predictions

    # Return frame
    return df


def fill_age(df):
    '''
    Whole process of the prediction of Age feature

    Parameters
    ----------
    df: object
        Pandas Frame

    '''
    # Prepare dataset for predicting Embarked
    train_emb = df.drop(['PassengerId', 'Cabin'], axis=1)

    # Prepare column list for converting categorical data to binaries
    categorical = train_emb.select_dtypes(include='category').columns
    # Create dummy variables for KNN Regression prediction
    train_emb = convert_categorical2Binary(train_emb, categorical)

    # Split data as X, y
    X_train_emb, y_train_emb, X_test_emb = split_X_y(train_emb, 'Age')

    # Predict the missing Embarked values
    predictions = regression_with_KNN(X_train_emb, y_train_emb, X_test_emb)

    # Add predictions to original frame
    df.loc[df.Age.isnull(), 'Age'] = predictions

    # Return frame
    return df


# Import frame
df = pd.read_csv('./data/kaggle-Titanic/train.csv')

# Print generic information about frame
gen_info(df)

# Process frame
df2 = process(df)

# Draw correlation matrix
draw_corr_matrix(df)

# Run ANOVA-test
# Define formula
f = 'Survived ~ Pclass*Sex*Age*SibSp*Parch'
table = anova(df, f, test='Chisq')
table

# Plan for filling missing values
df2.columns
list_missing_features_fraction(df2)  # Lists of features which have missing vals

# Embarked [Predicting]
df2 = fill_embarked(df2)

# Age [Predicting]
df2 = fill_age(df2)

# Cabin - [Drop]
df2 = df2.drop(['Drop'], axis=1)
