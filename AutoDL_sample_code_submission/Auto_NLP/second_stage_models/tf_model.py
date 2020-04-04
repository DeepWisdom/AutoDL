import tensorflow as tf 
import numpy as np

class FT_tf_model(object):
    def __init__(self, config):
        self.sequence_length = config['sequence_length']
        self.embedding_size = config['embedding_size']
        self.vocabulary_size = config['vocabulary_size']
        self.num_classes = config['num_classes']
        
        self.rr = 52
        self.add_rr = 20
        self.l2_reg_lambda = 0
        self.hidden_size = 128
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")

        self.ft_w_embed = self.get_token_embeddings(self.vocabulary_size, self.embedding_size, zero_pad = False, name = 'w_embed')
        
        
        with tf.variable_scope('FT', reuse=tf.AUTO_REUSE):
            embed1_word = tf.nn.embedding_lookup(self.ft_w_embed, self.input_x) 
            self.embed1_word = tf.expand_dims(embed1_word, -1)
            
            pooled_outputs = []
            filter_sizes = [2,3]
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_size, 1, 64]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[64]), name="b")
                    conv = tf.nn.conv2d(
                        self.embed1_word,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            
            num_filters_total = 64 * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            W1 = tf.get_variable(shape=[num_filters_total, 128], dtype=tf.float32, name='des_w1', initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(shape=[128], dtype=tf.float32, name='des_b1', initializer=tf.contrib.layers.xavier_initializer())
            x1 = tf.nn.xw_plus_b(self.h_pool_flat, W1, b1, name="dense1")
            x1 = tf.nn.relu(x1)
            
            W = tf.get_variable(shape=[128, self.num_classes], dtype=tf.float32, name='des_w', initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(shape=[self.num_classes], dtype=tf.float32, name='des_b', initializer=tf.contrib.layers.xavier_initializer())
            x = tf.nn.xw_plus_b(x1, W, b, name="dense")
            
            if self.num_classes == 2:
                self.probs = tf.nn.sigmoid(x)
                self.losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=self.input_y))
            else:
                self.probs = tf.nn.softmax(x)
                self.losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=self.input_y))
        
        
    def get_token_embeddings(self, vocab_size, num_units, zero_pad=True, name = 'shared_weight_matrix'):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            embeddings = tf.get_variable(name+'_tag_weight_mat', dtype=tf.float32, shape=(vocab_size, num_units), initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                embeddings = tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)
            
            return embeddings

    
    
    
    def fit(self, X_list, Y, epochs = 1, callbacks = None, verbose = 1, batch_size = 64, shuffle = True):
        
        index = [ i for i in range(len(Y))]
        np.random.shuffle(index)
        X = X_list[0]
        Y_b = np.eye(self.num_classes)[Y]
        X = X[index]
        Y_b = Y_b[index]
        
        if batch_size >= len(X):
            batch_size = int(len(X)/2)
        
        train_step = tf.train.AdamOptimizer(0.0025).minimize(self.losses)
        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        saver = tf.train.Saver() 
        with tf.Session() as sess:
            sess.run([init_global, init_local])
            rounds = min(int(len(X)/batch_size), self.rr)

            for i in range(rounds):
                start = i*batch_size
                end = (i+1)*batch_size
                _ = sess.run(train_step, feed_dict = {self.input_x:X[start:end], self.input_y:Y_b[start:end]})
                
            self.rr += self.add_rr
            saver.save(sess, 'ft.ckpt')
            
        return 0
            

    def predict(self, X, batch_size = 64, training = False):
        if batch_size >= len(X):
            batch_size = int(len(X)/2)
            
        rounds = int(len(X)/batch_size)
        saver = tf.train.Saver() 
        with tf.Session() as sess:
            saver.restore(sess, 'ft.ckpt')
            for i in range(rounds):
                start = i*batch_size
                end = (i+1)*batch_size 
                if i == 0:
                    probs = sess.run(self.probs, feed_dict = {self.input_x:X[start:end]})
                else:
                    p = sess.run(self.probs, feed_dict = {self.input_x:X[start:end]})
                    probs = np.concatenate((probs, p), axis = 0)
            if rounds*batch_size < len(X):
                start = rounds*batch_size
                p = sess.run(self.probs, feed_dict = {self.input_x:X[start:]})
                probs = np.concatenate((probs, p), axis = 0)
        if self.num_classes == 2:
            return probs[:,1]
        else:
            return probs
