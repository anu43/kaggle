# Import libraries
import matplotlib.pyplot as plt
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

# Import frame
df = pd.read_csv('../data/kaggle-Titanic/train.csv')

# Print generic information about frame
gen_info(df)

# Change types of features
# Fill NaN values as np.nan not to get error
df.fillna(np.nan, inplace=True)
# Set new types for certain features
types = {
        'PassengerId': 'category',
        'Survived': 'bool',
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
df.drop('Name', axis=1, inplace=True)

# Draw correlation matrix
draw_corr_matrix(df)
