import tensorflow as tf
import numpy as np
from collections import defaultdict
from load_sequence import Data_Factory
import pymysql
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability ")
tf.flags.DEFINE_string("object_path", "../blen80", "Dataset path")
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size ")
count =0
# 打开数据库连接
db = pymysql.connect(host='localhost',
                           port=3306,
                           user='root',
                           password='root',
                           database='bookrecommend',)
if __name__=='__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()
    #ckpt = tf.train.get_checkpoint_state('blen80_128')
    save_file='blen80_128'
    saver = tf.train.import_meta_graph('blen80_128.meta')
    df = Data_Factory(0.8, FLAGS.object_path + '/')
    with tf.Session() as sess:
        saver.restore(sess, save_file)
        graph = tf.get_default_graph()
        # 加载模型中的操作节点
        user_features = graph.get_operation_by_name('Placeholder').outputs[0]
        positive_features = graph.get_operation_by_name('Placeholder_1').outputs[0]
        positive_word = graph.get_operation_by_name('Placeholder_3').outputs[0]
        positive = graph.get_operation_by_name('AddN').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        train_phase = graph.get_operation_by_name('Placeholder_5').outputs[0]
        users_list = df.binded_users.keys()
        test_rating_map = df.user_positive_list_test
        test_predict_10 = defaultdict(set)
        One_features, One_words = [], []
        for X_it in df.All_data['X_item']:
            feature = []
            index = [0.0] * (df.item_field_M - df.item_bind_M)
            feature.append(X_it[0])
            feature.append(X_it[1])
            index[X_it[1] - df.item_bind_M] = 1.0
            words = X_it[2].strip().split()
            One_features.append(feature)
            One_words.append(list(map(float, words)))
        for user_key in users_list:
            count += 1
            if count == 101:
                break
            us = user_key.split(';')
            user_feature = [int(u) for u in us]
            One_users = [user_feature for i in range(df.item_bind_M)]
            One_history = []
            pred_re = []
            One_len = []
            ranks = ''
            batch = FLAGS.batch_size  # 200
            batches = df.item_bind_M // batch
            for i in range(batches + 1):
                t = batch * (i + 1) if batch * \
                                       (i + 1) < df.item_bind_M else df.item_bind_M
                feed_dict = {user_features: One_users[i * batch:t],
                             positive_features: One_features[i * batch:t],
                             positive_word: One_words[i * batch:t],
                             dropout_keep_prob: 1.0,
                             train_phase: False}
                pred_fm = sess.run(positive, feed_dict=feed_dict)
                pred_fm = np.reshape(pred_fm, -1)
                pred_re.extend(pred_fm)
            pred_re = np.reshape(pred_re, -1)
            pred_index = np.argsort(-pred_re)  # 排序，得分最高的在最前面
            pred_index = pred_index[:10]
            for pr in pred_index:
                item = df.item_map[pr]
                items = item.split(';')
                ranks+=items[0]+','
            cursor = db.cursor()
            sql = "INSERT INTO rank (stu_id, item_ids,stu_grade) VALUES ('%s', '%s', '%s')"%(us[0], ranks, us[2])
            try:
                cursor.execute(sql)
                db.commit()
            except:
                db.rollback()
            sql2 = "INSERT INTO user (stu_id, password,grade,school) VALUES ('%s', '%s', '%s', '%s')"%(us[0], us[0],us[2],us[1])
            try:
                cursor.execute(sql2)
                db.commit()
            except:
                db.rollback()
    db.close()
