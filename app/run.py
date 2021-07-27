import json
from os import name
import plotly
import pandas as pd
import re

import nltk
nltk.download(['punkt', 'wordnet'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """ Tokenise, standardise, lemmatize, and strip URLs from text

    Arguments:
        text (str): text to process

    Returns:
        clean_tokens (list): list of clean tokens
    """

    # Replace URLs in text with a standard placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Remove non-alphanumeric characters
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    
    # Tokenise, lemmatise, and convert to lower case
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()    
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesAndCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Create visuals
    # Define list of graphs and their layout properties
    graphs = [
        {
            'data': [], # Data populated in code below

            'layout': {
                'title': 'Message Genres by Category',
                'yaxis': {
                    'dtick': 1
                },
                'xaxis': {
                    'title': "Count"
                },
                'barmode': "stack",
                'height': 1000,
                'margin': {
                    'l': 200
                }
            }
        },
        {
            'data': [], # Data populated in code below

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Characters in message",
                    'range': [0, 500]
                }
            }
        }
    ]

    # Extract data for first graph and create Graph Objects
    # Count of each category, split by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_names = df.iloc[:, 4:].columns.str.replace('_', ' ')
    category_names = [category.title() for category in category_names]

    for genre in genre_names:
        category_counts = df[df['genre'] == genre].iloc[:, 4:].sum()
        graphs[0]['data'].append(Bar(
            name=genre.title(),
            y=category_names,
            x=category_counts,
            orientation= 'h'
        ))

    # Extract data for second graph and create Graph Objects
    # Histogram of message lengths
    message_lengths = df['message'].str.len()
    graphs[1]['data'].append(Histogram(
        x=message_lengths
    ))
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()