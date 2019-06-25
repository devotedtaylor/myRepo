import os
import time
import random
from load_sequence import Data_Factory
from tensorflow.python import debug as tf_debug
import tensorflow as tf
from collections import defaultdict
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability ")
tf.flags.DEFINE_string("object_path", "blen80", "Dataset path")
tf.flags.DEFINE_string("train_path", "train.txt", "Train data path")
tf.flags.DEFINE_string(
    "validation_path",
    "validation.txt",
    "Validation data path")
tf.flags.DEFINE_string("test_path", "test.txt", "Test data path")
tf.flags.DEFINE_boolean(
    "allow_soft_placement",
    True,
    "Allow device soft device placement")
tf.flags.DEFINE_boolean(
    "log_device_placement",
    False,
    "Log placement of ops on devices")

# Training parameters
tf.flags.DEFINE_string(
    'loss_type',
    'square_loss',
    'Specify a loss type (square_loss or log_loss).')
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 1200, "Number of training epochs ")
tf.flags.DEFINE_integer(
    'pretrain', -1, 'flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
tf.flags.DEFINE_string(
    'optimizer',
    'AdagradOptimizer',
    'Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
tf.flags.DEFINE_float('lamda', 0, 'Regularizer for bilinear part.')
tf.flags.DEFINE_float('lr', 0.1, 'Learning rate.')
tf.flags.DEFINE_integer(
    'verbose',
    1,
    'Show the results per X epochs (0, 1 ... any positive integer)')
tf.flags.DEFINE_integer('hidden_factor', 128, 'Number of hidden factors.')
tf.flags.DEFINE_integer(
    'batch_norm',
    0,
    'Whether to perform batch normaization (0 or 1)')


class DeepModel:
    def __init__(
            self,
            user_field_M,
            item_field_M,
            pretrain_flag,
            save_file,
            hidden_factor,
            epoch,
            batch_size,
            learning_rate,
            lamda_bilinear,
            keep,
            optimizer_type,
            batch_norm,
            verbose,
            random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        # performance of each epoch
        self.rec = 0

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        tf.set_random_seed(self.random_seed)
        # input data
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        self.user_features = tf.placeholder(tf.int32, shape=[None, None])
        self.positive_features = tf.placeholder(tf.int32, shape=[None, None])
        self.negative_features = tf.placeholder(tf.int32, shape=[None, None])
        self.clc_num = tf.placeholder(tf.float32, shape=[None, self.item_field_M-df.item_bind_M])
        self.positive_word = tf.placeholder(tf.float32, shape=[None, 128])
        self.negative_word = tf.placeholder(tf.float32, shape=[None, 128])
        self.train_phase = tf.placeholder(tf.bool)

        # Variables.
        self.weights = self._initialize_weights()
        # _________ sum_square part for positive (u,i)_____________
        self.user_feature_embeddings = tf.nn.embedding_lookup(
            self.weights['user_feature_embeddings'], self.user_features)
        self.positive_item_embeddings = tf.nn.embedding_lookup(
            self.weights['item_feature_embeddings'], self.positive_features)
        self.negative_item_embeddings = tf.nn.embedding_lookup(
            self.weights['item_feature_embeddings'], self.negative_features)

        self.clc_list = tf.convert_to_tensor(
            df.school_list, dtype=tf.int32)  # [1*570]

        self.positive_item_school = tf.gather(
            self.clc_list, tf.reshape(tf.squeeze(self.user_features[:, 1]), [-1]))
        self.positive_item_school_emb = tf.nn.embedding_lookup(
            self.weights['item_feature_embeddings'],
            self.positive_item_school)  # [None,256]

        self.positive_item_features = tf.concat(
            [self.positive_item_embeddings, tf.expand_dims(self.positive_word,1)], axis=1)
        self.negative_item_features = tf.concat(
            [self.negative_item_embeddings, tf.expand_dims(self.negative_word,1)], axis=1)

        self.user_embeddings = tf.concat([self.user_feature_embeddings[:,1,:],
                                                 self.user_feature_embeddings[:,2,:]],axis=-1)

        self.positive_item = tf.concat(
            [self.user_feature_embeddings,self.positive_item_embeddings], axis=1)

        self.negative_item = tf.concat(
            [self.user_feature_embeddings,self.negative_item_embeddings], axis=1)
        self.user_bias = tf.reduce_sum(
            tf.nn.embedding_lookup(
                self.weights['user_feature_bias'],
                self.user_features),
            1)  # None * 1
        self.din_i_p = tf.concat([self.user_embeddings, self.positive_item_embeddings[:,1,:]], axis=-1)  # [B, 192]
        for i in range(0, len(self.layers)):
            self.din_i_p = tf.add(tf.matmul(self.din_i_p,self.weights['layer_%d' % i]), self.weights['bias_%d' % i])  # None * layer[i] * 1
            if self.batch_norm:
                self.din_i_p = self.batch_norm_layer(
                    self.din_i_p, train_phase=self.train_phase, scope_bn='bn_%d' %
                    i)  # None * layer[i] * 1
            self.din_i_p = tf.sigmoid(self.din_i_p)
            # dropout at each Deep layer
            self.din_i_p = tf.nn.dropout(self.din_i_p, self.dropout_keep_prob)
        self.din_i_p = tf.matmul(self.din_i_p, self.weights['prediction'])  # None * 1

        self.din_i_n = tf.concat([self.user_embeddings, self.negative_item_embeddings[:,1,:]], axis=-1)  # [B, 192]
        for i in range(0, len(self.layers)):
            self.din_i_n = tf.add(tf.matmul(self.din_i_n,
                                            self.weights['layer_%d' % i]),
                                  self.weights['bias_%d' % i])  # None * layer[i] * 1
            if self.batch_norm:
                self.din_i_n = self.batch_norm_layer(
                    self.din_i_n, train_phase=self.train_phase, scope_bn='bn_%d' %
                                                                         i)  # None * layer[i] * 1
            self.din_i_n = tf.sigmoid(self.din_i_n)
            # dropout at each Deep layer
            self.din_i_n = tf.nn.dropout(self.din_i_n, self.dropout_keep_prob)
        self.din_i_n = tf.matmul(self.din_i_n, self.weights['prediction'])  # None * 1

        pos_element_wise_product_list = []
        pos_count = 0
        for i in range(0, 5):
            for j in range(i + 1, 5):
                pos_element_wise_product_list.append(
                    tf.multiply(self.positive_item[:, i, :], self.positive_item[:, j, :]))
                pos_count += 1
        self.pos_element_wise_product = tf.stack(pos_element_wise_product_list)  # (M'*(M'-1)) * None * K
        self.pos_element_wise_product = tf.transpose(self.pos_element_wise_product, perm=[1, 0, 2],
                                                 name="pos_element_wise_product")  # None * (M'*(M'-1)) * K

        self.FM = tf.reduce_sum(self.pos_element_wise_product, 1, name="afm")  # None * K
        if self.batch_norm:
            self.FM = self.batch_norm_layer(
                self.FM, train_phase=self.train_phase, scope_bn='bn_fm_po')
        # dropout at the FM layer
        self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)
        if self.batch_norm:
            self.FM = self.batch_norm_layer(
                self.FM, train_phase=self.train_phase, scope_bn='bn_fm_po')
        # dropout at the FM layer
        self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)
        Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
        self.positive_item_bias = tf.reduce_sum(
            tf.nn.embedding_lookup(
                self.weights['item_feature_bias'],
                self.positive_features),
            1)  # None * 1
        self.positive = self.din_i_p + tf.add_n(
            [Bilinear, self.user_bias, self.positive_item_bias])  # None * 1

        neg_element_wise_product_list = []
        neg_count = 0
        for i in range(0, 5):
            for j in range(i + 1, 5):
                neg_element_wise_product_list.append(
                    tf.multiply(self.negative_item[:, i, :], self.negative_item[:, j, :]))
                neg_count += 1
        self.neg_element_wise_product = tf.stack(neg_element_wise_product_list)  # (M'*(M'-1)) * None * K
        self.neg_element_wise_product = tf.transpose(self.neg_element_wise_product, perm=[1, 0, 2],
                                                 name="neg_element_wise_product")  # None * (M'*(M'-1)) * K

        self.FM_negative = tf.reduce_sum(self.neg_element_wise_product, 1, name="nfm")  # None * K
        if self.batch_norm:
            self.FM_negative = self.batch_norm_layer(
                self.FM_negative, train_phase=self.train_phase, scope_bn='bn_fm_ne')
        self.FM_negative = tf.nn.dropout(
            self.FM_negative,
            self.dropout_keep_prob)  # dropout at the FM layer
        # _________out _________
        Bilinear_negative = tf.reduce_sum(
            self.FM_negative, 1, keep_dims=True)  # None * 1
        self.negative_item_bias = tf.reduce_sum(
            tf.nn.embedding_lookup(
                self.weights['item_feature_bias'],
                self.negative_features),
            1)  # None * 1

        self.negative = self.din_i_n + tf.add_n(
            [Bilinear_negative, self.user_bias, self.negative_item_bias])  # None * 1

        # Compute the loss.
        self.loss = -tf.log(tf.sigmoid(self.positive - self.negative))
        # self.loss = tf.log(1 + tf.exp(-tf.sigmoid(self.positive - self.negative)))
        self.loss = tf.reduce_sum(self.loss)
        # self._loss = tf.add(self.loss, tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
        #                     name='objective')
        # Optimizer.
        if self.optimizer_type == 'AdamOptimizer':
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == 'AdagradOptimizer':
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate,
                initial_accumulator_value=0.1).minimize(self.loss)
        elif self.optimizer_type == 'GradientDescentOptimizer':
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)
        elif self.optimizer_type == 'MomentumOptimizer':
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=0.95).minimize(
                self.loss)
        # init
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        #self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        self.sess.run(init)
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name(
                'user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name(
                'item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name(
                'user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name(
                'item_feature_bias:0')
            # bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub, ib = sess.run(
                    [user_feature_embeddings, item_feature_embeddings, user_feature_bias, item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(
                ue, trainable=False, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(
                ie, trainable=False, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(
                ub, trainable=False, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(
                ib, trainable=False, dtype=tf.float32)
            print("load!")
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(tf.random_uniform(
                [self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')  # user_field_M * 1
            all_weights['item_feature_bias'] = tf.Variable(tf.random_uniform(
                [self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')  # item_field_M * 1

            self.layers = [128,64]
            glorot = np.sqrt(2.0 / (self.hidden_factor*3 + self.layers[0]))
            num_layer = 2
            all_weights['layer_0'] = tf.Variable(
                np.random.normal(
                    loc=0,
                    scale=glorot,
                    size=(
                        self.hidden_factor*3,
                        self.layers[0])),
                dtype=np.float32)
            all_weights['bias_0'] = tf.Variable(
                np.random.normal(
                    loc=0,
                    scale=glorot,
                    size=(
                        1,
                        self.layers[0])),
                dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])),
                    dtype=np.float32)  # 1 * layer[i]
                # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                dtype=np.float32)  # layers[-1] * 1
            # # 1 * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(
            x,
            decay=0.9,
            center=True,
            scale=True,
            updates_collections=None,
            is_training=True,
            reuse=None,
            trainable=True,
            scope=scope_bn)
        bn_inference = batch_norm(
            x,
            decay=0.9,
            center=True,
            scale=True,
            updates_collections=None,
            is_training=False,
            reuse=True,
            trainable=True,
            scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        positive_feature = []
        positive_word = []
        negative_feature = []
        negative_word = []
        for X_po in data['X_positive']:
            feature = []
            feature.append(X_po[0])
            feature.append(X_po[1])
            words = X_po[2].strip().split()
            positive_feature.append(feature)
            positive_word.append(list(map(float, words)))

        for X_ne in data['X_negative']:
            feature = []
            feature.append(X_ne[0])
            feature.append(X_ne[1])
            words = X_ne[2].strip().split()
            negative_feature.append(feature)
            negative_word.append(list(map(float, words)))

        feed_dict = {
            self.user_features: data['X_user'],
            self.positive_features: positive_feature,
            self.positive_word: positive_word,
            self.clc_num: data['X_history'],
            self.negative_features: negative_feature,
            self.negative_word: negative_word,
            self.dropout_keep_prob: FLAGS.dropout_keep_prob,
            self.train_phase: True}
        loss, opt = self.sess.run(
            (self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def train(self, Train_data, Test_data):  # fit a dataset
        # Check Init performance
        lastLoss = 100000000000
        #model.evaluate()
        for epoch in range(self.epoch):
            total_loss = 0
            t1 = time.time()
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(
                    Train_data, self.batch_size)
                # Fit training
                loss = self.partial_fit(batch_xs)
                total_loss = total_loss + loss
            t2 = time.time()
            print(
                "the total loss in %d th iteration is: %f cost [%.1f s]" %
                (epoch, total_loss, t2 - t1))
            if abs(lastLoss - total_loss) < 1:
                print("converge!")
                # break;
            lastLoss = total_loss
            if (epoch + 1) % 600 == 0:
                start_time = time.time()
                print("epoch:%f" % epoch)
                model.evaluate()
                end_time = time.time()
                print("Evaluation cost [%.1f s]" % (end_time - start_time))
        print("end train begin save")
        if self.pretrain_flag < 0:
            print("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    # generate a random block of training data
    def get_random_block_from_data(self, train_data, batch_size):
        X_user, X_positive, X_negative, X_history = [], [], [], []
        X_negative_clc = []
        X_positive_clc = []
        X_len = []
        all_items = df.binded_items.values()
        # get sample
        while len(X_user) < batch_size*1:
            index = np.random.randint(0, len(train_data['X_user']))
            user_index = train_data['X_user'][index][0]
            # uniform sampler
            user_features = ";".join(
                [str(item) for item in train_data['X_user'][index][0:]])
            user_id = df.binded_users[user_features]  # get userID
            # get positive list for the userID
            pos = df.user_positive_list[user_id]
            # uniform sample a negative itemID from negative set
            neg = np.random.randint(len(all_items))
            neg_list = []
            pos_clc_list = [0.0]*(self.item_field_M-df.item_bind_M)
            pos_clc_list[train_data['X_item'][index][1]-df.item_bind_M]=1.0
            neg_clc_list = [0.0]*(self.item_field_M-df.item_bind_M)
            history = [0]*(self.item_field_M-df.item_bind_M)
            his_str = df.clc_number[user_features].strip().split()
            for i in range(len(history)):
                history[i]=float(his_str[i].split(':')[1])
            for i in range(1):
                while neg in pos or neg in neg_list:
                    neg = np.random.randint(len(all_items))
                neg_list.append(neg)
                negative_feature = df.item_map[neg].strip().split(
                    ';')  # get negative item feature
                nf = []
                neg_clc_list[int(negative_feature[1])-df.item_bind_M] = 1.0
                nf.append(int(negative_feature[0]))
                nf.append(int(negative_feature[1]))
                nf.append(negative_feature[2])
                X_positive_clc.append(pos_clc_list)
                X_history.append(history)
                X_negative.append(nf)
                X_negative_clc.append(neg_clc_list)
                X_user.append(train_data['X_user'][index])
                X_positive.append(train_data['X_item'][index])
        return {
            'X_user': X_user,
            'X_positive': X_positive,
            'X_negative': X_negative,
            'X_history': X_history,
            'X_positive_clc': X_positive_clc,
            'X_negative_clc': X_negative_clc
        }

    def evaluate(self):  # evaluate the results for an input set
        users_list = df.binded_users.keys()
        test_rating_map = df.user_positive_list_test
        test_predict_5 = defaultdict(set)
        test_predict_10 = defaultdict(set)
        hitsum = 0
        test_sum = 0
        count = 0
        user_K = 0
        One_features, One_words = [], []
        One_index = []
        for X_it in df.All_data['X_item']:
            feature = []
            index = [0.0]*(self.item_field_M-df.item_bind_M)
            feature.append(X_it[0])
            feature.append(X_it[1])
            index[X_it[1]-df.item_bind_M]=1.0
            words = X_it[2].strip().split()
            One_features.append(feature)
            One_words.append(list(map(float, words)))
        for user_key in users_list:
            us = user_key.split(';')
            user_feature = [int(u) for u in us]
            One_users = [user_feature for i in range(df.item_bind_M)]
            One_history = []
            pred_re = []
            One_len=[]
            history = [0]*(self.item_field_M-df.item_bind_M)
            his_str = df.clc_number[user_key].strip().split()
            for i in range(len(history)):
                history[i]=float(his_str[i].split(':')[1])
            for i in range(df.item_bind_M):
                One_history.append(history)
                One_len.append(self.item_field_M-df.item_bind_M)
            batch = FLAGS.batch_size #200
            batches = df.item_bind_M//batch
            for i in range(batches+1):
                t = batch * (i + 1) if batch * \
                    (i + 1) < df.item_bind_M else df.item_bind_M
                feed_dict = {self.user_features: One_users[i * batch:t],
                             self.positive_features: One_features[i * batch:t],
                             self.positive_word: One_words[i * batch:t],
                             self.clc_num: One_history[i * batch:t],
                             self.dropout_keep_prob: 1.0,
                             self.train_phase: False}
                pred_fm = self.sess.run(self.positive, feed_dict=feed_dict)
                pred_fm = np.reshape(pred_fm, -1)
                pred_re.extend(pred_fm)
            pred_re = np.reshape(pred_re, -1)
            pred_index = np.argsort(-pred_re)  # 排序，得分最高的在最前面
            for i in range(len(pred_index)):
                if count == 5:
                    count = 0
                    break
                if df.binded_users[user_key] in df.user_positive_list:
                    if not df.user_positive_list[df.binded_users[user_key]].__contains__(
                            pred_index[i]):
                        count += 1
                        test_predict_5[df.binded_users[user_key]].add(pred_index[i])
                else:
                    count += 1
                    test_predict_5[df.binded_users[user_key]].add(pred_index[i])
            for i in range(len(pred_index)):
                if count == 10:
                    count = 0
                    break
                if df.binded_users[user_key] in df.user_positive_list:
                    if not df.user_positive_list[df.binded_users[user_key]].__contains__(
                            pred_index[i]):
                        count += 1
                        test_predict_10[df.binded_users[user_key]].add(pred_index[i])
                else:
                    count += 1
                    test_predict_10[df.binded_users[user_key]].add(pred_index[i])
            # if df.binded_users[user_key] in test_rating_map:
            #     test_sum += len(test_rating_map[df.binded_users[user_key]])
            #     user_K += 5
        for key in df.user_positive_list:
            if key in test_rating_map:
                test_sum+=len(test_rating_map[key])
                user_K += 5
                for value in test_rating_map[key]:
                    if value in test_predict_5[key]:
                        hitsum += 1
        rec = hitsum * 1.0 / test_sum
        precision = hitsum * 1.0 / user_K
        print('Rec@5: ' + str(rec) + 'Pre@5: ' + str(precision))
        self.rec_prec(test_predict_5,test_rating_map,5)
        self.rec_prec(test_predict_10, test_rating_map, 5)
        self.rec_prec(test_predict_10, test_rating_map, 10)

    def rec_prec(self, predict, test, K):
        test_sum, user_K = 0, 0
        hitsum = 0
        for key in test:
            test_sum += len(test[key])
            user_K += K
            values = sorted(list(predict[key]), reverse=True)
            values = values[:K]
            for value in test[key]:
                if values.__contains__(value):
                    hitsum += 1
        rec = hitsum * 1.0 / test_sum
        precision = hitsum * 1.0 / user_K
        print('Rec@%d: ' % K + str(rec) + ' Pre@%d: ' % K + str(precision))
        return rec

if __name__ == '__main__':
    # Data loading
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()
    df = Data_Factory(0.8, FLAGS.object_path + '/')
    if FLAGS.verbose > 0:
        print(
            "EFM: dataset=%s, factors=%d, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d" %
            (FLAGS.object_path,
             FLAGS.hidden_factor,
             FLAGS.loss_type,
             FLAGS.num_epochs,
             FLAGS.batch_size,
             FLAGS.lr,
             FLAGS.lamda,
             FLAGS.dropout_keep_prob,
             FLAGS.optimizer,
             FLAGS.batch_norm))

    save_file = 'pretrain/%s_%d' % (FLAGS.object_path, FLAGS.hidden_factor)
    # Training
    t1 = time.time()
    model = DeepModel(
        df.user_field_M,
        df.item_field_M,
        FLAGS.pretrain,
        save_file,
        FLAGS.hidden_factor,
        FLAGS.num_epochs,
        FLAGS.batch_size,
        FLAGS.lr,
        FLAGS.lamda,
        FLAGS.dropout_keep_prob,
        FLAGS.optimizer,
        FLAGS.batch_norm,
        FLAGS.verbose)
    print("begin train")
    model.train(df.Train_data, df.Test_data)
    print("end train")
    print("finish")
