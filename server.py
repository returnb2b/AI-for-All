# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import datetime
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

app = Flask(__name__)

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

#저장된 모델 불러오는 모델 초기화
saver = tf.train.Saver()
model = tf.global_variables_initializer()

sess = tf.Session()
sess.run(model)

save_path = "./model/saved.cpkt"  # cpkt파일 불러오기
saver.restore(sess, save_path)

@app.route("/", methods=['GET', 'POST'])
def index():
	if request.method == 'GET':
		return render_template('index.html')
	if request.method == 'POST':
		avg_temp = float(request.form['avg_temp'])
		min_temp = float(request.form['min_temp'])
		max_temp = float(request.form['max_temp'])
		rain_fall = float(request.form['rain_fall'])
		
	price = 0
	
	data = ((avg_temp, min_temp, max_temp, rain_fall), )  # 기존의 학습된 데이터와 같은 2차원 배열 만들기
	arr = np.array(data, dtype=np.float32)
	
	# 예측 수행
	x_data = arr[0:4]  # avg_temp, min_temp, max_temp, rain_fall
	dict = sess.run(hypothesis, feed_dict={X: x_data})
	
	price = dict[0]
	return render_template('index.html', price = price)

if __name__ == '__main__':
	app.run(debug=True)
