from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import openai
import os
import logging
from services.openai_service import generate_response

app = Flask(__name__)
CORS(app)


# Routes


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    text = data.get('data')
    passphrase = data.get('passphrase')
    if passphrase != 'قوي':
        return jsonify({"message": 'your not authorized', "response": False})
    user_input = text

    try:
        output = generate_response(user_input,'','')
        print(output)
        return jsonify({"response": True, "message": output["output_text"]})
    except Exception as e:
        print(e)
        error_message = f'Error: {str(e)}'
        return jsonify({"message": error_message, "response": False})


# Functions

if __name__ == '__main__':
    logging.info("Flask app started")
    app.run(host='0.0.0.0', port=4000)
