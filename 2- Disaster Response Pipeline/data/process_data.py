# Run the file by the following
# cd 2- Disaster Response Pipeline
# python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

from dataclasses import replace
import sys
import pandas as pd
import numpy as np
import sqlalchemy as db


def load_data(messages_filepath, categories_filepath):
    """
    Description:
        Load Data from the csv files

    Args:
        messages_filepath: The filepath of the csv file containing actual tweet messages
        categories_filepath: The filepath of the csv file containing message categories

    Return:
        df: dataframe containing messages and categoreis
    """

    # Load messages and categories
    messages = pd.read_csv('data/disaster_messages.csv')
    categories = pd.read_csv('data/disaster_categories.csv')
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    """
    Description:
        Cleans the data in the raw merged dataframe of messages and categories into
        a processable dataframe ready to export

        Args:
            df: The raw dataframe containign messages and categories

        Return:
            df: The cleaned dataframe ready to be loaded to a database    
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", n=-1, expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # now lets get rid of last two characters by slicing the
    # string using lambda function
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the 'categories' column'
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # drop the original 'categories' column from our dataframe `df`
    if 'categories' in df.columns:
        df.drop('categories', axis='columns', inplace=True)

    # remove the non-binary values from the 'related' column
    categories = categories[categories.related != 2]

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle Null Values
    # -- drop the 'original' column since it has a lot of null values but no usable data for our scenario
    del df['original']
    # -- drop the null values
    df = df.dropna()

    # return the dataframe
    return df


def save_data(df, database_filename):
    """
    Description:
        Load the provided dataframe into a new sqlite database with the name provided 
        into the table named 'messages'

    Args:
        df: The dataframe containing data to be loaded
        database_filename: The name of the database file

    Return:
        Void
    """

    # Create the database file for next steps and export
    engine = db.create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
