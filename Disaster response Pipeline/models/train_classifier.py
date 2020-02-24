import sys
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle
nltk.download(['punkt', 'wordnet'])
import os
import re


def load_data(database_filepath):
    
    '''
    Load Data from databse
    
    Args:
    database_filepath: path to SQL database
    
    Returns:
    X: features DataFrame
    y: label DataFrame
    category_names: Label names
    
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name='disasters', con=engine)
    X = df['message']
    y = df[df.columns[5:]]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    
    '''
    Tokenize function to process the text data
    
    Args:
    text: list of text messages.
        
    Returns:
    clean_tokens: tokenized text for Machine learning modeling.
    
    '''
    
    reg = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(reg, text)
    
    for url in detected_urls:
        text = text.replace(url, "placeholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    processed_tokens= []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        processed_tokens.append(clean_token)

    return processed_tokens


def build_model():
    '''
    Build model and improve the model with GridSearchCV
    
    Returns:
    Trained model after performing grid search to find better parameters
    
    '''
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10, min_samples_split = 10))),
    ])
    
    parameters = {'clf__estimator__max_depth': [30, 100, None]}

    model = GridSearchCV(pipeline, parameters)

    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluates the model and shows accuracy, precision, and recall of the tuned mode.
    
    Args:
    model : ML Pipeline and tuned model
    X_test: Features for test data
    Y_test: Labels for test data
    '''
    
    y_preds = model.predict(X_test)
    
    results = {}

    for pred, label, col in zip(y_preds.transpose(), Y_test.values.transpose(), Y_test.columns):
        print(col)
        print(classification_report(label, pred))
        results[col] = classification_report(label, pred)
        
    performance_metrics = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    
    for col in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[col], y_preds[:,num], average='weighted')
        performance_metrics.set_value(num+1, 'Category', col)
        performance_metrics.set_value(num+1, 'f_score', f_score)
        performance_metrics.set_value(num+1, 'precision', precision)
        performance_metrics.set_value(num+1, 'recall', recall)
        num += 1
        
    print('Mean f_score:', performance_metrics['f_score'].mean())
    print('Mean precision:', performance_metrics['precision'].mean())
    print('Mean recall:', performance_metrics['recall'].mean())
    
    
def save_model(model, model_filepath):
    
    '''
    Saves the model and dumps into a pickle file
    
    Args:
    model: tuned and ML pipelined model
    model_filepath: file path to save the model.    
    '''

    pickle.dump(model, open(model_filepath, 'wb'))
    
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()