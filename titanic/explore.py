# Import libraries
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
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


def convert_categorical2Binary(df):
    pass


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
        'Survived': 'int8',
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

# Cabin [Not decided]

# Age [Predicting]

# Embarked [Predicting]
