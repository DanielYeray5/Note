from flask import Flask, render_template, send_file
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
from graphviz import Source
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import f1_score, recall_score

app = Flask(__name__)

# Cargar y preparar los datos
df = pd.read_csv('/Users/danielyeraynogueziniestra/Downloads/x.csv')

# Verificar si la columna 'Socioeconomic Score' existe
if 'Socioeconomic Score' not in df.columns:
    raise ValueError("La columna 'Socioeconomic Score' no existe en el dataset")

# Verificar si las columnas 'Sleep Hours' y 'Study Hours' existen
if 'Sleep Hours' not in df.columns:
    df['Sleep Hours'] = np.random.uniform(-0.5, 2, len(df))  # Crear la columna con valores aleatorios en el rango [-0.5, 2]

if 'Study Hours' not in df.columns:
    df['Study Hours'] = np.random.uniform(-0.5, 2, len(df))  # Crear la columna con valores aleatorios en el rango [-0.5, 2]

# Convertir 'Socioeconomic Score' en categorías discretas
df['Socioeconomic Score'] = pd.cut(df['Socioeconomic Score'], bins=3, labels=["low", "medium", "high"])

X = df.copy()
X_train = X[['Sleep Hours', 'Study Hours']]
y_train = X['Socioeconomic Score']

# Entrenar el modelo
clf_tree_reduced = DecisionTreeClassifier(max_depth=2, random_state=42)
clf_tree_reduced.fit(X_train.values, y_train)

# Calcular el F1 score y el recall
y_pred = clf_tree_reduced.predict(X_train.values)
f1 = f1_score(y_train, y_pred, average='weighted')
recall = recall_score(y_train, y_pred, average='weighted')

def plot_decision_boundary(clf, X, y, plot_training=True, resolution=1000):
    mins = X.min(axis=0) - 1
    maxs = X.max(axis=0) + 1
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
    plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="low")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="medium")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="high")
        plt.axis([mins[0], maxs[0], mins[1], maxs[1]])               
    plt.xlabel('Sleep Hours', fontsize=14)
    plt.ylabel('Study Hours', fontsize=14, rotation=90)
    plt.legend()

@app.route('/')
def index():
    return render_template('index.html', f1_score=f1, recall=recall)

@app.route('/plot')
def plot():
    plt.figure(figsize=(12, 6))
    plot_decision_boundary(clf_tree_reduced, X_train.values, y_train)
    plt.savefig('static/decision_boundary.png')
    plt.close()  # Cierra la figura para evitar advertencias
    return send_file('static/decision_boundary.png', mimetype='image/png')

@app.route('/tree')
def tree():
    export_graphviz(
        clf_tree_reduced,
        out_file="static/socioeconomic_tree.dot",
        feature_names=X_train.columns,
        class_names=["low", "medium", "high"],
        rounded=True,
        filled=True
    )
    with open("static/socioeconomic_tree.dot") as f:
        dot_graph = f.read()
    source = Source(dot_graph)
    source.render("static/socioeconomic_tree", format="png")
    return send_file('static/socioeconomic_tree.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8000)