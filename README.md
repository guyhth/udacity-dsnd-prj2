# Disaster Response Pipeline Project

## Introduction

### Background

TODO: Description of the project and objectives

### The data

TODO: describe model imbalance

### The web app

TODO: description and screenshots

## Instructions

### Installation

Python 3.x and the following libraries are required:
- Data analysis and machine learning: NumPy, Scikit-Learn, NLTK
- Database and files: SQLalchemy, Pickle
- Web app and visualisation: Flask, Plotly

### Files

- `data` directory
    - `disaster_messages.csv` and `disaster_categories.csv`: Raw data files for processing from Figure Eight.
    - `process_data.py`: Runs the Extract, Transform, Load (ETL) pipeline to extract, clean and merge the data from the CSV files, and save it to an SQLite database.
- `models` directory
    - `train_classifier.py`: Loads the data from the SQLite database and runs a machine learning pipeline to train a model for classifying messages. Optionally, the model parameters can be optimised using GridSearchCV. The model is saved as a file for use by the web app.
- `app` directory
    - `run.py`: Launches the Flask web app.
    - `templates` directory: HTML/Jinja templates for the Flask web app.

### Running the code

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements

Thank you to [Figure Eight](https://www.figure-eight.com/) for the dataset, and to [Udacity](https://www.udacity.com) for the excellent Data Scientist Nanodegree programme.

## Licence

This software is provided under the [MIT licence](https://opensource.org/licenses/MIT).