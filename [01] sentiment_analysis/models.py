import tensorflow as tf

class logistic_regression:
    def __init__(self, sess, vocab_size, n_class=2, lr=1e-2):
        self.sess = sess
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.lr = lr
        self._build_net()
        
    def _build_net(self):
        self.input_x = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
        self.input_y = tf.placeholder(tf.int32, shape=(None,))
        Y_one_hot = tf.one_hot(self.input_y, self.n_class)
        
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', dtype=tf.float32,
                               initializer=tf.truncated_normal((self.vocab_size, self.n_class)))
            b = tf.get_variable('b', dtype=tf.float32,
                               initializer=tf.constant(0.1, shape=(self.n_class,)))
            logits = tf.nn.xw_plus_b(self.input_x, W, b)
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)
            self.prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            
            
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
        
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            
        with tf.variable_scope("accuracy"):
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, input_x, input_y):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x})
    
    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    
class three_layer_net:
    def __init__(self, sess, vocab_size, hidden_size=128, n_class=2, lr=1e-2):
        self.sess = sess
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.lr = lr
        self._build_net()
        
    def _build_net(self):
        with tf.variable_scope("placeholder"):
            self.input_x = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
            self.input_y = tf.placeholder(tf.int32, shape=(None,))
        
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            W1 = tf.get_variable("W1", dtype=tf.float32,
                                 initializer=tf.truncated_normal((self.vocab_size, self.hidden_size)))
            b1 = tf.get_variable("b1", dtype=tf.float32,
                                initializer=tf.constant(0.1, shape=(self.hidden_size,)))
            W2 = tf.get_variable("W2", dtype=tf.float32,
                                initializer=tf.truncated_normal((self.hidden_size, self.n_class)))
            b2 = tf.get_variable("b2", dtype=tf.float32,
                                initializer=tf.constant(0.1, shape=(self.n_class,)))
            h = tf.nn.relu(tf.nn.xw_plus_b(self.input_x, W1, b1))
            logits = tf.nn.xw_plus_b(h, W2, b2)
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)
            self.prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
        
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

        with tf.variable_scope("accuracy"):
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, input_x, input_y):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x})
    
    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    
class RNN:

    def __init__(self, sess, vocab_size, embedding_size=300, hidden_size=128, n_class=2, lr=1e-2):
        self.sess = sess
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.lr = lr
        self._build_net()

    def _build_net(self):
        with tf.variable_scope("placeholder"):
            self.input_x = tf.placeholder(tf.int32, (None, None))
            self.input_y = tf.placeholder(tf.int32, (None,))
            input_length = tf.reduce_sum(tf.sign(self.input_x), axis=1)
        
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', dtype=tf.float32,
                initializer=tf.random_uniform((self.vocab_size, self.embedding_size), minval=-1.0, maxval=1.0))
            embedded_input_x = tf.nn.embedding_lookup(W, self.input_x)
            
        with tf.variable_scope("recurrent"):
            cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)
            _, states = tf.nn.dynamic_rnn(cell,
                                          embedded_input_x, 
                                          input_length, 
                                          dtype=tf.float32)
        
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):        
            W = tf.get_variable('W', dtype=tf.float32,
                               initializer=tf.truncated_normal((self.hidden_size, self.n_class)))
            b = tf.get_variable('b', dtype=tf.float32,
                               initializer=tf.constant(0.1, shape=(self.n_class,)))
            logits = tf.nn.xw_plus_b(states, W, b)
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)
            self.prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
            
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):        
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, 
                                                       global_step, 
                                                       1e+3, 
                                                       0.9, 
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        
        with tf.variable_scope("accuracy"):        
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, input_x, input_y):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x})
    
    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y})
    

class LSTM:

    def __init__(self, sess, vocab_size, embedding_size=300, hidden_size=128, n_class=2, lr=1e-2, trainable=True):
        self.sess = sess
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.lr = lr
        self.trainable = trainable
        self._build_net()

    def _build_net(self):
        with tf.variable_scope("placeholder"):
            self.input_x = tf.placeholder(tf.int32, (None, None))
            self.input_y = tf.placeholder(tf.int32, (None,))
            self.embedding_placeholder = tf.placeholder(tf.float32, (self.vocab_size, self.embedding_size))
            input_length = tf.reduce_sum(tf.sign(self.input_x), axis=1)
        
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', dtype=tf.float32,
                initializer=tf.random_uniform((self.vocab_size, self.embedding_size), minval=-1.0, maxval=1.0), trainable=self.trainable)
            self.embedding_init = W.assign(self.embedding_placeholder)
            embedded_input_x = tf.nn.embedding_lookup(W, self.input_x)
            
        with tf.variable_scope("recurrent"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            _, states = tf.nn.dynamic_rnn(cell,
                                          embedded_input_x, 
                                          input_length, 
                                          dtype=tf.float32)

        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):        
            W = tf.get_variable('W', dtype=tf.float32,
                               initializer=tf.truncated_normal((self.hidden_size, self.n_class)))
            b = tf.get_variable('b', dtype=tf.float32,
                               initializer=tf.constant(0.1, shape=(self.n_class,)))
            logits = tf.nn.xw_plus_b(states.h, W, b)
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)
            self.prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
            
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):        
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, 
                                                       global_step, 
                                                       1e+3, 
                                                       0.9, 
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        
        with tf.variable_scope("accuracy"):        
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
            
    def embedding_assign(self, embedding):
        return self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})

    def train(self, input_x, input_y):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x})
    
    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y})


class LSTM_onehot:

    def __init__(self, sess, vocab_size, hidden_size=128, n_class=2, lr=1e-2, trainable=True):
        self.sess = sess
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.lr = lr
        self.trainable = trainable
        self._build_net()

    def _build_net(self):
        with tf.variable_scope("placeholder"):
            self.input_x = tf.placeholder(tf.int32, (None, None))
            self.input_y = tf.placeholder(tf.int32, (None,))
            input_length = tf.reduce_sum(tf.sign(self.input_x), axis=1)

        with tf.variable_scope("onehot_encoding", reuse=tf.AUTO_REUSE):
            onehot_input_x = tf.one_hot(self.input_x, self.vocab_size)

        with tf.variable_scope("recurrent"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            _, states = tf.nn.dynamic_rnn(cell,
                                          onehot_input_x,
                                          input_length,
                                          dtype=tf.float32)

        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', dtype=tf.float32,
                                initializer=tf.truncated_normal((self.hidden_size, self.n_class)))
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=tf.constant(0.1, shape=(self.n_class,)))
            logits = tf.nn.xw_plus_b(states.h, W, b)
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)
            self.prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr,
                                                       global_step,
                                                       1e+3,
                                                       0.9,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

        with tf.variable_scope("accuracy"):
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.sess.run(tf.global_variables_initializer())

    def train(self, input_x, input_y):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y})

    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x})

    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y})
    
class biLSTM:

    def __init__(self, sess, vocab_size, embedding_size=300, hidden_size=128, n_class=2, lr=1e-2):
        self.sess = sess
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.lr = lr
        self._build_net()

    def _build_net(self):
        with tf.variable_scope("placeholder"):
            self.input_x = tf.placeholder(tf.int32, (None, None))
            self.input_y = tf.placeholder(tf.int32, (None,))
            input_length = tf.reduce_sum(tf.sign(self.input_x), axis=1)
        
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', dtype=tf.float32,
                initializer=tf.random_uniform((self.vocab_size, self.embedding_size), minval=-1.0, maxval=1.0))
            embedded_input_x = tf.nn.embedding_lookup(W, self.input_x)
        
        with tf.variable_scope("recurrent"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            ((_, _),
             (fw_states, bw_states)) = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                       bw_cell,
                                                                       embedded_input_x, 
                                                                       input_length, 
                                                                       dtype=tf.float32)

            states = tf.concat((fw_states.h, bw_states.h), 1)
            
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):        
            W = tf.get_variable('W', dtype=tf.float32,
                               initializer=tf.truncated_normal((2*self.hidden_size, self.n_class)))
            b = tf.get_variable('b', dtype=tf.float32,
                               initializer=tf.constant(0.1, shape=(self.n_class,)))
            logits = tf.nn.xw_plus_b(states, W, b)
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)
            self.prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
            
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):        
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, 
                                                       global_step, 
                                                       1e+3, 
                                                       0.9, 
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        
        with tf.variable_scope("accuracy"):        
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            
        self.sess.run(tf.global_variables_initializer())

    def train(self, input_x, input_y):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x})
    
    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    
    
class deepBiLSTM:

    def __init__(self, sess, vocab_size, embedding_size=300, hidden_size=128, n_class=2, lr=1e-2):
        self.sess = sess
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.lr = lr
        self._build_net()

    def _build_net(self):
        with tf.variable_scope("placeholder"):
            self.input_x = tf.placeholder(tf.int32, (None, None))
            self.input_y = tf.placeholder(tf.int32, (None,))
            self.dropout_keep_prob = tf.placeholder(tf.float32)
            input_length = tf.reduce_sum(tf.sign(self.input_x), axis=1)
        
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', dtype=tf.float32,
                initializer=tf.random_uniform((self.vocab_size, self.embedding_size), minval=-1.0, maxval=1.0))
            embedded_input_x = tf.nn.embedding_lookup(W, self.input_x)

            
        with tf.variable_scope("recurrent"):
            fw_multi_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size),
                input_keep_prob=self.dropout_keep_prob,
                output_keep_prob=self.dropout_keep_prob) for _ in range(3)])
            bw_multi_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size),
                input_keep_prob=self.dropout_keep_prob,
                output_keep_prob=self.dropout_keep_prob) for _ in range(3)])
            ((_, _), (fw_states, bw_states)) = tf.nn.bidirectional_dynamic_rnn(fw_multi_cell,
                                                                               bw_multi_cell,
                                                                               embedded_input_x, 
                                                                               input_length, 
                                                                               dtype=tf.float32)
            states = tf.concat([tf.concat([fw_states[i].h, bw_states[i].h], 1) for i in [0,1,2]], 1)
            
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):        
            W = tf.get_variable('W', dtype=tf.float32,
                               initializer=tf.truncated_normal((6*self.hidden_size, self.n_class)))
            b = tf.get_variable('b', dtype=tf.float32,
                               initializer=tf.constant(0.1, shape=(self.n_class,)))
            logits = tf.nn.xw_plus_b(states, W, b)
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)
            self.prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
            
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):        
            optimizer = tf.train.AdamOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)
        
        with tf.variable_scope("accuracy"):        
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, input_x, input_y, dropout_keep_prob=0.7):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y, self.dropout_keep_prob: dropout_keep_prob})
    
    def predict(self, input_x, dropout_keep_prob=1.0):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x, self.dropout_keep_prob: dropout_keep_prob})
    
    def get_accuracy(self, input_x, input_y, dropout_keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y, self.dropout_keep_prob: dropout_keep_prob})    
    
    
class GRU:

    def __init__(self, sess, vocab_size, embedding_size=300, hidden_size=128, n_class=2, lr=1e-2):
        self.sess = sess
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.lr = lr
        self._build_net()

    def _build_net(self):
        with tf.variable_scope("placeholder"):
            self.input_x = tf.placeholder(tf.int32, (None, None))
            self.input_y = tf.placeholder(tf.int32, (None,))
            input_length = tf.reduce_sum(tf.sign(self.input_x), axis=1)
            
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', dtype=tf.float32,
                initializer=tf.random_uniform((self.vocab_size, self.embedding_size), minval=-1.0, maxval=1.0))
            embedded_input_x = tf.nn.embedding_lookup(W, self.input_x)
        
        with tf.variable_scope("recurrent"):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            _, states = tf.nn.dynamic_rnn(cell,
                                          embedded_input_x, 
                                          input_length, 
                                          dtype=tf.float32)
            
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):        
            W = tf.get_variable('W', dtype=tf.float32,
                               initializer=tf.truncated_normal((self.hidden_size, self.n_class)))
            b = tf.get_variable('b', dtype=tf.float32,
                               initializer=tf.constant(0.1, shape=(self.n_class,)))
            logits = tf.nn.xw_plus_b(states, W, b)
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1)
            self.prediction = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
            
        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):      
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, 
                                                       global_step, 
                                                       1e+3, 
                                                       0.9, 
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        
        with tf.variable_scope("accuracy"):        
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        
    def train(self, input_x, input_y):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y})
    
    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x})
    
    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y})


class CNN(object):
    """
    The implementation is based on following:
    dennybritz: simplified implementation of Kim's Convolutional Neural Networks for Sentence Classification paper in Tensorflow.
    """
    def __init__(
      self, sess, vocab_size, sequence_length=30, embedding_size=300,
            filter_sizes=(3,4,5), num_filters=128, n_class=2, lr=1e-2, trainable=True):
        self.sess = sess
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.n_class = n_class
        self.lr = lr
        self.trainable = trainable
        self._build_net()

    def _build_net(self):
        # Placeholders for input, output
        with tf.variable_scope("placeholder"):
            self.input_x = tf.placeholder(tf.int32, (None, self.sequence_length))
            self.input_y = tf.placeholder(tf.int32, (None,))
            self.dropout_keep_prob = tf.placeholder(tf.float32)
            self.embedding_placeholder = tf.placeholder(tf.float32, (self.vocab_size, self.embedding_size))
        # Embedding layer
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                initializer = tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), trainable=self.trainable)
            self.embedding_init = W.assign(self.embedding_placeholder)
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in self.filter_sizes:
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=tf.AUTO_REUSE):
                # Convolution Layer
                filter_shape = (filter_size, self.embedding_size, 1, self.num_filters)
                W = tf.get_variable("W", dtype=tf.float32,
                    initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable("b", dtype=tf.float32,
                    initializer=tf.constant(0.1, shape=(self.num_filters,)))
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, (-1, num_filters_total))

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                "W",
                shape=(num_filters_total, self.n_class),
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=(self.n_class,)), name="b")
            logits = tf.nn.xw_plus_b(h_drop, W, b, name="logits")
            self.prob = tf.reduce_max(tf.nn.softmax(logits), axis=1, name="prob")
            self.prediction = tf.cast(tf.argmax(logits, 1), tf.int32, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, 
                                                       global_step, 
                                                       1e+3, 
                                                       0.9, 
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct = tf.equal(self.prediction, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        
        self.sess.run(tf.global_variables_initializer())
        
    def embedding_assign(self, embedding):
        return self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})
    
    def train(self, input_x, input_y, dropout_keep_prob=0.7):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y, self.dropout_keep_prob: dropout_keep_prob})

    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x, self.dropout_keep_prob: 1.0})

    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y, self.dropout_keep_prob: 1.0})
