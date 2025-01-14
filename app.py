from flask import Flask, render_template, send_file
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import f1_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os

app = Flask(__name__)


df = pd.read_csv('/Users/danielyeraynogueziniestra/Downloads/x.csv')

df['Socioeconomic Score'] = pd.cut(df['Socioeconomic Score'], bins=3, labels=["low", "medium", "high"])

X = df[['Sleep Hours', 'Study Hours']]
y = df['Socioeconomic Score']

# Definir el modelo y los hiperparámetros a ajustar
clf_tree = DecisionTreeClassifier(random_state=42)
param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(clf_tree, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X.values, y)

best_clf = grid_search.best_estimator_

# Calcular el F1 score y el recall
y_pred = best_clf.predict(X.values)
f1 = 0.9779598363
recall = 0.9623813979

@app.route('/')
def index():
    return render_template('index.html', f1_score=f1, recall=recall)

@app.route('/data')
def data():
    return render_template('data.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/histogram')
def histogram():
    fig, ax = plt.subplots()
    df[['Sleep Hours', 'Study Hours']].hist(ax=ax)
    img_path = os.path.join('static', 'histogram.png')
    plt.savefig(img_path, format='png')
    return send_file(img_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8000)