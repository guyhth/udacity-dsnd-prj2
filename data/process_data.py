# Import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Load messages and categories datasets and merge into a single DataFrame

    Arguments:
        messages_filepath (str): path to messages dataset (CSV file)
        categories_filepath (str): path to categories dataset (CSV file)

    Returns:
        df (DataFrame): merged Pandas DataFrame
    
    """
    # Load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets using inner join to only keep rows that appear in both datasets
    df = pd.merge(messages, categories, on='id', how='inner')

    return df


def clean_data(df):
    """ Clean the data by splitting, cleaning, and converting the categories column

    Arguments:
        df (DataFrame): the data to process

    Returns:
        df (DataFrame): cleaned data
    
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    first_row = categories.iloc[0]

    # Use this row to create new column names for categories
    category_colnames = first_row.apply(lambda col_string: col_string[0:-2])
    categories.columns = category_colnames

    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Convert the 'related' column to binary (contains 0, 1, 2 by default)
    categories['related'] = categories['related'].astype('str').str.replace('2', '1')
    categories['related'] = pd.to_numeric(categories['related'])

    # Drop the original categories column from df
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # Check for and remove duplicate messages
    num_duplicates = df.shape[0] - df['message'].nunique()
    df.drop_duplicates(subset='message', inplace=True)
    print("{} duplicate messages removed.".format(num_duplicates))

    return df


def save_data(df, database_filename):
    """ Save data to an SQLite database. Table will be replaced if it already exists.

    Arguments:
        df (DataFrame): data to save
        database_filename (string): filename to save data to
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessagesAndCategories', engine, index=False, if_exists='replace')


def main():
    # Expect 4 arguments, otherwise print the usage guide
    if len(sys.argv) == 4:

        # Pull filepaths from the command line arguments provided by the user 
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()