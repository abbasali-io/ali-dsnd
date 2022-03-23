# Disaster Response Pipeline Project

<br/><br/>

## Project Description

In this Project, I created an ETL pipeline for data transformation and a Machine Learning pipeline to classify the messages into multiple classes. The pipeline build a model which is used in a Flask application. The application is deployed at servers to be used by the disaster management staff, by putting in a tweeted message and getting categories that can be related to the message. Moving forward the model can be part of an automated pipeline to detect the required categories of tweets automatically and respond to the emergency workers as required.

**Disaster Application Screenshot**

<!-- ![](https://raw.githubusercontent.com/abbasali-io/ali-dsnd/main/2-%20Disaster%20Response%20Pipeline/Disaster_Response_Screenshot.png) -->
<img src="https://raw.githubusercontent.com/abbasali-io/ali-dsnd/main/2-%20Disaster%20Response%20Pipeline/Disaster_Response_Screenshot.png" alt="Screenshot" width="60%"/>
<br/><br/>
## Project Assets

The project uses following key assets;

`./notebooks/ETL Pipeline Preperation.ipynb` <- _The Data Experiments Notebook_

`./notebooks/Disaster_Response_ML_Pipeline_Preperation.ipynb` <- _The Machine Learning Model Experiments Notebook_

`./data/process_data.py` <- _Use this file to generate database file_

`./model/train_classifier.py` <- _Use this file to generate machine learning model_

`./app/templates/\*.html`<- _Use this folder files to manipulate the html_

`./app/run.py` <- _Use this file to run the program_
<br/><br/>

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

   - **To run ETL pipeline that cleans data and stores in database**

     - `cd 2- Disaster Response Pipeline`

     - `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

   - **To run ML pipeline that trains classifier and saves**

     - `cd 2- Disaster Response Pipeline`

     - `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

   - `cd 2- Disaster Response Pipeline`

   - `cd app`

   - `python run.py`

3. Go to http://localhost:3001/
