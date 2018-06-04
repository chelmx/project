import tensorflow as tf
from tensorflow.contrib import rnn

class myLstmClassifier():
    def __init__(self, input_size, time_size, layer_size, hidden_size, output_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, [None, time_size, input_size])
            self.Y = tf.placeholder(tf.float32, [None, output_size])

            lstm_cell = rnn.BasicLSTMCell(num_units = hidden_size, forget_bias = 1.0)
            mlstm_cell = rnn.MultiRNNCell([lstm_cell for i in range(layer_size)])

            lstm_output, lstm_state = tf.nn.dynamic_rnn(mlstm_cell, self.X, dtype = tf.float32)
            h_state = lstm_output[:, -1, :]

            weight = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev = 0.1), dtype = tf.float32)
            bais = tf.Variable(tf.constant(0.1,shape = [output_size]), dtype = tf.float32)

            self.output = tf.matmul(h_state, weight) + bais

        summary_writer = tf.summary.FileWriter('./logs', self.graph)

mlc = myLstmClassifier(5, 30, 1, 16, 2)
