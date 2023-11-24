import json

import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import requests
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from model_predict import load_data, load_model_params, predict

app = Flask(__name__)

print("init model...")
param = load_model_params()
print("init model done")

# 处理GET请求，返回表单以输入X
@app.route('/')
def home():
    return render_template('DataInput.html')


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()  # 获取用户输入的数据
    # print(data)
    df = pd.DataFrame(data)  # 将数据转换为DataFrame
    prediction = predict(df, param)  # 使用模型进行预测
    return jsonify({'prediction': prediction.tolist()})  # 将预测结果返回给用户



@app.route("/pred", methods=["POST"])
def pred():
    data = request.form.to_dict()
    df = pd.DataFrame(data, index=[0])
    print(df)
    prediction = predict(df, param)
    print(prediction)
    return redirect(url_for('res', prediction=prediction))

@app.route("/result", methods=["GET"])
def res(prediction=[0]):
    return render_template('result.html', prediction=prediction[0])

# 启动应用程序
if __name__ == '__main__':
    app.run(debug=True)
