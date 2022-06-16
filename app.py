from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/classify' , methods=['POST'])
def classify():

    sc = StandardScaler()
    inp = [float(x) for x in request.form.values()]
    inp = [np.array(inp)]
    minp = sc.fit_transform(inp)
    res = model.predict(minp)

    return render_template('index.html', cl_out="It seems to be {}".format("Malignant" if res[0]==1 else "Benign"))

if __name__ == '__main__':
    app.run(debug=True)