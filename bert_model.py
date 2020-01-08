#!/usr/bin/python
# coding=utf8

"""
# Created : 2018/12/28
# Version : python2.7
# Author  : yibo.li 
# File    : bert_model.py
# Desc    : 
"""

import os
import tensorflow as tf
import numpy as np
from datetime import datetime

from bert import modeling
from bert import optimization
from bert.data_loader import *

from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix,accuracy_score

import logging
import sys

log_dir = 'logdir'
log_file = os.path.join(log_dir, 'log.txt')
if not os.path.isdir(log_dir):
  os.makedirs(log_dir)

logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file))
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

processors = {"zwtt": ZWTTProcessor}#修改data_loader.py

class BertModel():
    def __init__(self, bert_config, num_labels, seq_length, init_checkpoint):
        self.bert_config = bert_config
        self.num_labels = num_labels
        self.seq_length = seq_length

        self.input_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        #self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learn_rate')

        self.model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        self.inference()

    def inference(self):

        output_layer = self.model.get_pooled_output()
        logging.info(output_layer)
        with tf.variable_scope("loss"):
            def apply_dropout_last_layer(output_layer):
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
                return output_layer

            def not_apply_dropout(output_layer):
                return output_layer

            output_layer = tf.cond(self.is_training, lambda: apply_dropout_last_layer(output_layer),
                                   lambda: not_apply_dropout(output_layer))

            match_1 = tf.strided_slice(output_layer, [0], [train_batch_size], [2])
            match_2 = tf.strided_slice(output_layer, [1], [train_batch_size], [2])

            match = tf.concat([match_1, match_2], 1)

            self.logits = tf.layers.dense(match, self.num_labels, name='fc')

            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")
            logging.info(self.y_pred_cls)


            self.r_labels = tf.strided_slice(self.labels, [0], [train_batch_size], [2])
            print(self.r_labels)

            one_hot_labels = tf.one_hot(self.r_labels, depth=self.num_labels, dtype=tf.float32)
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_labels)
            #self.loss = tf.reduce_mean(cross_entropy, name="loss")

            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            per_example_loss =  - (30*one_hot_labels[:,0] * log_probs[:,0]) \
                                - (9*one_hot_labels[:,1] * log_probs[:,1]) \
                                - (2*one_hot_labels[:,2] * log_probs[:,2]) \
                                - (2*one_hot_labels[:,3] * log_probs[:,3]) \
                                - (9*one_hot_labels[:,4] * log_probs[:,4]) \
                                + 1e-10

            self.loss = tf.reduce_mean(per_example_loss)

            self.optim = optimization.create_optimizer(self.loss, learning_rate, num_train_steps, num_warmup_steps, False)

            #self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")
            print(self.acc)
            self.cm = tf.contrib.metrics.confusion_matrix(tf.argmax(one_hot_labels, 1), self.y_pred_cls, num_classes=num_labels)
            print(self.cm)
def make_tf_record(output_dir, data_dir, vocab_file):
    tf.gfile.MakeDirs(output_dir)       #"model/bert"
    processor = processors[task_name]() #"atec"
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    train_file = os.path.join(output_dir, "train.tf_record")
    eval_file = os.path.join(output_dir, "eval.tf_record")

    # save data to tf_record
    if not os.path.isfile(train_file):
        train_examples = processor.get_train_examples(data_dir)
        file_based_convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, train_file)
        del train_examples
    # eval data
    if not os.path.isfile(eval_file):
        eval_examples = processor.get_dev_examples(data_dir)
        file_based_convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, eval_file)
        del eval_examples


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def read_data(data, batch_size, is_training, num_epochs):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.

    if is_training:
        data = data.shuffle(buffer_size=500000)
        data = data.repeat(num_epochs)


    data = data.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))
    return data


def get_test_example():
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    # save data to tf_record
    #test_examples = processor.get_test_examples("")#测试数据目录
    test_examples = processor.get_test_examples(data_dir)
    features = get_test_features(test_examples, label_list, max_seq_length, tokenizer)

    return features


def evaluate(sess, model):
    """
    评估 val data 的准确率和损失
    """
    # dev data
    test_record = tf.data.TFRecordDataset("./model/bert/eval.tf_record")
    test_data = read_data(test_record, train_batch_size, False, 1)
    test_iterator = test_data.make_one_shot_iterator()
    test_batch = test_iterator.get_next()

    ds_cm = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    ds_pred = []
    ds_label = []
    while True:
        try:
            features = sess.run(test_batch)
            feed_dict = {model.input_ids: features["input_ids"],
                         model.input_mask: features["input_mask"],
                         model.segment_ids: features["segment_ids"],
                         model.labels: features["label_ids"],
                         model.is_training: False}
            d_cm,d_pred,d_labels = sess.run([model.cm,model.y_pred_cls,model.r_labels], feed_dict=feed_dict)
            ds_cm += d_cm
            ds_pred.extend(d_pred)
            ds_label.extend(d_labels)
        except Exception as e:
            #print("Dev error:")
            #print(e)
            break

    ds_acc = (ds_cm[0][0] + ds_cm[1][1] + ds_cm[2][2] + ds_cm[3][3] + ds_cm[4][4]) / \
                np.sum(ds_cm,axis=(0,1))

    return ds_acc,ds_cm,ds_pred,ds_label

def rongcuodu(y_true_list, y_pred_list):
    sum_ = []
    num = 0
    num_dict = {0:0, 1:0, 2:0, 3:0, 4:0}
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        sum_.append(abs(y_true - y_pred))
    for i in sum_:
        num_dict[i] += 1
    for k, v in num_dict.items():
        num_dict[k] = v / len(y_true_list)
    return num_dict

def main():
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    with tf.Graph().as_default():
        # train data
        train_record = tf.data.TFRecordDataset("./model/bert/train.tf_record")
        train_data = read_data(train_record, train_batch_size, True, num_train_epochs)
        train_iterator = train_data.make_one_shot_iterator()

        model = BertModel(bert_config, num_labels, max_seq_length, init_checkpoint)
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=30)
        train_steps = 0
        val_loss = 0.0
        val_acc = 0.0
        best_rho_val = 0.0
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            train_batch = train_iterator.get_next()
            while True:
                try:
                    train_steps += 1
                    features = sess.run(train_batch)
                    feed_dict = {model.input_ids: features["input_ids"],
                                 model.input_mask: features["input_mask"],
                                 model.segment_ids: features["segment_ids"],
                                 model.labels: features["label_ids"],
                                 model.is_training: True}
                    _, train_loss, train_acc,train_cm,train_pred,train_labels = \
                    sess.run([model.optim, model.loss, model.acc, model.cm,model.y_pred_cls,model.r_labels],
                        feed_dict=feed_dict)
                    if train_steps % 10 == 0:
                        logging.info("Train:global_nums:%d,loss:%f,acc:%f,rho:%f" % \
                            (train_steps,train_loss,train_acc,pearsonr(train_labels, train_pred)[0]))

                    if train_steps % 100 == 0:
                        val_acc,val_cm,val_pred,val_label = evaluate(sess, model)
                        val_rho = pearsonr(val_label, val_pred)[0]
                        logging.info("Dev:global_nums:%d,acc:%f,rho:%f" % \
                            (train_steps,val_acc,val_rho))
                        logging.info("容错：")
                        logging.info(rongcuodu(val_label, val_pred))
                        logging.info(val_cm)

                        #if val_rho > best_rho_val:
                            # 保存最好结果
                            #best_rho_val = val_rho
                        path = saver.save(sess, "./model/bert/model", global_step=train_steps)
                        logging.info("save model:%s" % path)
                except Exception as e:
                    #logging.info("train error:")
                    #logging.info(e)
                    break


def test_model(sess, graph, features):
    """

    :param sess:
    :param graph:
    :param features:
    :return:
    """
    input_ids = graph.get_operation_by_name('input_ids').outputs[0]
    input_mask = graph.get_operation_by_name('input_mask').outputs[0]
    segment_ids = graph.get_operation_by_name('segment_ids').outputs[0]
    labels = graph.get_operation_by_name('labels').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]

    pred = graph.get_operation_by_name('loss/pred').outputs[0]
    r_labels = graph.get_operation_by_name('loss/StridedSlice_2').outputs[0]

    data_len = len(features)
    batch_size = 20
    num_batch = int((len(features) - 1) / batch_size) + 1 #巧妙

    test_y = []
    label_y = []
    for i in range(num_batch):
        #logging.info(i)
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)

        batch_len = end_index-start_index

        _input_ids = np.array([data.input_ids for data in features[start_index:end_index]])
        _input_mask = np.array([data.input_mask for data in features[start_index:end_index]])
        _segment_ids = np.array([data.segment_ids for data in features[start_index:end_index]])
        _labels = np.array([data.label_id for data in features[start_index:end_index]])
        feed_dict = {input_ids: _input_ids,
                     input_mask: _input_mask,
                     segment_ids: _segment_ids,
                     labels: _labels,
                     is_training: False}
        pred_y,r_label_y = sess.run([pred,r_labels], feed_dict=feed_dict)

        test_y.extend(pred_y)
        label_y.extend(r_label_y)
    #logging.info(test_y)
    return test_y,label_y



def test():
    features = get_test_example()
    model_path = "./model/bert"
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_path)
    selected_ckpt = ckpt.all_model_checkpoint_paths[-4]

    saver = tf.train.import_meta_graph(selected_ckpt+'.meta', graph=graph)
    
    #saver.restore(sess, tf.train.latest_checkpoint(model_path))
    saver.restore(sess,selected_ckpt)


    test_y,label_y = test_model(sess, graph, features)
    logging.info(len(test_y),len(label_y))
    #bad case analyzes
    logging.info(test_y)
    logging.info(label_y)

    logging.info("Test:acc:%f,rho:%f" % (accuracy_score(label_y, test_y),pearsonr(label_y, test_y)[0]))

    logging.info("容错：")
    logging.info(rongcuodu(label_y, test_y))
    logging.info("confusion_matrix:")
    cm = confusion_matrix(label_y, test_y)
    logging.info(cm)
    '''

    test_df = pd.read_csv(sys.argv[1], sep='\t', header=None, names=["index", "s1", "s2"])
    
    test_index = np.array(test_df["index"])

    with open(sys.argv[2], 'w') as fout:
        for index,pre in enumerate(test_y):
            if pre >=0.5:
                fout.write(str(test_index[index]) + '\t1\n')
            else:
                fout.write(str(test_index[index]) + '\t0\n')
    '''
if __name__ == "__main__":
    data_dir = "data/zwtt"
    output_dir = "model/bert"
    task_name = "zwtt"
    vocab_file = "./bert/english_model/vocab.txt"
    bert_config_file = "./bert/english_model/bert_config.json"
    init_checkpoint = "./bert/english_model/bert_model.ckpt"
    max_seq_length = 380
    learning_rate = 2e-5
    train_batch_size = 20
    num_train_epochs = 3
    warmup_proportion = 0.1
    num_train_steps = int(27000 / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    num_labels = 5
    make_tf_record(output_dir, data_dir, vocab_file)
    main()
    test()

