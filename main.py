from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/SentAnalyzer')
def SentAnalyzer():
    return render_template('SentAnalyzer.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/policy')
def policy():
    return render_template('policy.html')


@app.route('/Analyze', methods=['POST'])
def predict(pipeline=None):
    global accuracy
    df = pd.read_csv("PreprocessedDataset.csv")
    df["clean_tweet"] = df["clean_tweet"].astype('U')
    df['label'] = df['label'].map({'Positive': 4, 'Neutral': 2, 'Negative': 0})
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('classifier', LinearSVC()),
    ])

    text_train, text_test, label_train, label_test = train_test_split(df['clean_tweet'], df['label'], test_size=0.3)

    pipeline.fit(text_train, label_train)
    pipeline.score(text_train, label_train)


    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = pipeline.predict(data)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
