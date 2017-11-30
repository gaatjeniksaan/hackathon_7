import sys
from flask import Flask
from flask import jsonify
from flask_restful import Api
from flask_restful import Resource
from flask_restful import reqparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer

from api_classes import OneHotDataFramer

app = Flask(__name__)
subset = ['has_analytics', 'fb_published', 'has_logo', 'name_length', 'show_map', 'num_payouts']
parser = reqparse.RequestParser()
for col in subset:
	try:
		parser.add_argument(col, type=int)
	except:
		raise ValueError('Did not pass the proper data for {}'.format(col))

def load_model(model_dir, model_name):
    return joblib.load(model_dir+'/'+model_name) 

model = load_model('.', 'model.pkl')

@app.route('/api/v1/predict', methods=['GET', "POST"])
def show_predictions():
    args = parser.parse_args()
    #print(args)
    #print('Your arguments are {}'.format(args))
    df = pd.DataFrame([args], columns=args.keys())
    #print(df.info())
    #print(df)
    prediction = model.predict(df)
    print('my first prediction {}'.format(prediction[0]))

    return jsonify(prediction[0])


if __name__ == '__main__':
    app.run(debug=True, port=8080)


