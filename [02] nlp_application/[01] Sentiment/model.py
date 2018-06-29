import tensorflow as tf

class CNN(object):
    """
    The implementation is based on following:
    dennybritz: simplified implementation of Kim's Convolutional Neural Networks for Sentence Classification paper in Tensorflow.
    """

    def __init__(
            self, sess, vocab_size, sequence_length=30, embedding_size=300,
            filter_sizes=(3, 4, 5), num_filters=128, n_class=2, lr=1e-2, trainable=True):
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
            self.embedding_placeholder = tf.placeholder(tf.float32, (self.vocab_size, self.embedding_size))
        # Embedding layer
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            W = tf.get_variable("W", dtype=tf.float32,
                                initializer=tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                trainable=self.trainable)
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

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            W = tf.get_variable(
                "W",
                shape=(num_filters_total, self.n_class),
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=(self.n_class,)), name="b")
            logits = tf.nn.xw_plus_b(h_pool_flat, W, b, name="logits")
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

    def train(self, input_x, input_y):
        return self.sess.run([self.loss, self.train_op], feed_dict={self.input_x: input_x, self.input_y: input_y})

    def predict(self, input_x):
        return self.sess.run((self.prediction, self.prob), feed_dict={self.input_x: input_x})

    def get_accuracy(self, input_x, input_y):
        return self.sess.run(self.accuracy, feed_dict={self.input_x: input_x, self.input_y: input_y})