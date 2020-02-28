# Disaster Response Pipeline Project

### Description

In this project, data science skills are used to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The Project uses a data set containing real messages that were sent during disaster events. This data set is used to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

 The aim of the project is to build a Natural Language Processing(NLP) tool that would categorize messages.The project is broken down into three sections as mentioned below.
 
    1.Process the data - Using ETL Pipeline to extract data from source, transform data and load them in database.             
    2.Train the data - Using Machine Learning Pipeline to train a model to be able to classify text message in categories.
    3.Create a Web App - Shows the results.
    
### About the Files 

In the data and models folder there are two ipynb - jupyter notebook files that helps in understanding the step by step procedure of categorize the messages.

ETL Preparation Notebook(data/process_data.py): Has all the information and steps about the implementation of ETL pipeline.
ML Pipeline Preparation Notebook(models/train_classifier.py): Has the implementation of Machine Learning Pipeline developed with NLTK and Scikit-Learn.
app/run.py file : This file used to launch the Flask web app to classify disaster messages.


### Instructions on Execution:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
