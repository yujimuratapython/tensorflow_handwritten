import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_loader

mnist = mnist_loader.read_data_sets("MNIST_data/", one_hot=True)

#トレーニング用画像データを最初の10個だけ表示
plt.imshow(mnist.train.images[:10].reshape((280,28)))
plt.gray()

#トレーニング用ラベルデータを最初の10個だけ表示
print(mnist.train.labels[:10])

# 初期値の設定
# 入力画像（28pixel*28pixel)
image_size = 28*28
# 出力値
output_num = 10
# 学習率
learning_rate = 0.001
# バッチの試行回数
loop_num = 30000
# バッチ1つに対する画像の枚数
batch_size = 100

#入力値
x = tf.placeholder("float",[None,784])
#重み
W = tf.Variable(tf.zeros([image_size, output_num]))
#バイアス
b = tf.Variable(tf.zeros([output_num]))
#ソフトマックス回帰式
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 出力データ (予測値) 
y_ = tf.placeholder(tf.float32, [None, 10])

# コスト関数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#最適化関数
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# 予測値と正解ラベルが同じ値であるかを確かめる
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 予測精度計算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#テスト用データセットの実行
print("test_accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

test_accuracy: 0.926