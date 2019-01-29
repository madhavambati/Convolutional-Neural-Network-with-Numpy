from flask import Flask, jsonify, render_template, request, flash, logging, session, redirect, url_for
import numpy as np
import sys
import pickle
import json
#sys.path.insert(0, 'MNIST')

from model.functions import predict
from preprocessing import * 

app = Flask(__name__)
@app.route('/')
def index():
	return render_template('index.html')
	#render_template('templates/index.html')

@app.route('/digit_process', methods=['POST'])
def digit_process():
	if(request.method == "POST"):
		img = request.get_json()
		img = preprocessing(img)

		save_path = 'model/params.pkl'
		params, cost = pickle.load(open(save_path, 'rb'))
		

		[f1, f2, w3, w4, b1, b2, b3, b4] = params
		digit, probability = predict(img, params)
		#print(digit, "%0.2f"%probability)

		#l = int(digit)
		#p = float(probability)

		data = { "digit":int(digit), "probability":float(np.round(probability, 3)) }
		data_json = json.dumps(data)
        
        
		return jsonify(data_json)
		print(done)

if __name__ == "__main__":
	app.run(debug=True)
