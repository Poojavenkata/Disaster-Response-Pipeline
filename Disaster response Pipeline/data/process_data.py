import sys
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Takes inputs as two CSV files and imports them as pandas dataframe.
    Merges the two dataframe into a single dataframe.
    
    Args:
    messages_file_path : messages CSV file
    categories_file_path : categories CSV file
    
    Returns:
    df : Final Dataframe after merging two input dataframes.
    data
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    
    return df




def clean_data(df):
    '''
    Takes inputs as the merged dataframe and cleans it so that it can 
    be used by Machine learning pipeline.
    
    Args:
    
    df dataframe: Merged dataframe from load function.

    
    Returns:
     df dataframe : Dataframe after performing all the clean operations.
    '''
    
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda s: s[:len(s)-2])
    for column in categories:
        categories[column] = categories[column].str.slice(start=-1)
        categories[column] = categories[column].astype(np.int)
    df = pd.concat([df, categories], axis=1)
    print("Total number of duplicate rows is {} out of {}".format(df.duplicated().sum(), df.shape[0]))
    df=df.drop_duplicates()
    print("Total number of duplicate rows is {} out of {}".format(df.duplicated().sum(), df.shape[0]))
    return df



def save_data(df, database_filename):
    
    '''
    Saves cleaned data to an SQL database. This data can be
    used by ML pipeline.
    
    Args:
    df dataframe: Cleaned data from the previous 'clean_data' function.
    database_file_name: File path of SQL Database into which the cleaned 
    data is to be saved
    
    Returns: None
    
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disasters', engine, index=False)  


    
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()