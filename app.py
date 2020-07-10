#coding: utf-8

from flask import Flask, request, jsonify, render_template
from extractor import Extractor


app = Flask(__name__)
extractor = Extractor()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def extract():
    '''
    For rendering results on HTML GUI
    '''
    raw_content = request.form.get('news')
    print(raw_content)
    result = extractor.find_opinions(raw_content)
    return render_template('index.html', extracts=result)


@app.route('/predict_api',methods=['POST'])
def extract_api():
    raw_content = request.get_json(force=True)
    result = extractor.find_opinions(raw_content.values())
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8999, debug=True)