import os
import json
import random
import tensorflow as tf
from transformers import TFGPT2LMHeadModel
# from models import GPT2Model
from transformers.modeling_tf_utils import shape_list
import collections
from tqdm import tqdm
import re

from config import basic_config
from preprocess_data import preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def check_data(config=None):
    if not config:
        config = basic_config()
    # 检查训练文件是否存在
    file_names = ['train.json', 'dev.json', 'test.json']
    reprocess_data = False
    for file_name in file_names:
        check_file_path = os.path.join(config.data_save_path, file_name)
        if not os.path.isfile(check_file_path):
            reprocess_data = True
            break
    if reprocess_data:
        print("处理原始数据")
        preprocess(config)
    return file_names


def to_tf_record(config):
    file_names = check_data(config)

    for file in file_names:
        split_name = file.replace(".json", '')
        print("将{}数据转化为tfrecord".format(split_name))
        raw_file = os.path.join(config.data_save_path, file)
        save_file = os.path.join(config.data_save_path,
                                 "{}.tfrecord".format(split_name))

        writer = tf.io.TFRecordWriter(save_file)
        with open(raw_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
        num = 0
        for line in tqdm(lines):
            num += 1
            if num > 1000:
                break
            sent_ids = json.loads(line)
            input_ids = sent_ids[:-1]  # label与inpu_ids错开一个
            label_ids = [-100] * len(input_ids)  # 只对最后一个数字求loss，其余为0
            label_ids[-1] = sent_ids[-1]

            input_ids = input_ids + [0] * (config.max_length - len(input_ids))
            label_ids = label_ids + [-100
                                     ] * (config.max_length - len(label_ids))
            assert len(input_ids) == config.max_length
            assert len(label_ids) == config.max_length

            features = {
                'input_ids':
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=list(input_ids))),
                'label_ids':
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=list(label_ids)))
            }
            tf_example = tf.train.Example(features=tf.train.Features(
                feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()


def to_tf_dataset(tfrecord_file,
                  batch_size,
                  is_training=False,
                  max_length=None):
    def _parse_fn(record):
        # print(record)
        features = {
            'input_ids': tf.io.FixedLenFeature([max_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([max_length], tf.int64)
        }
        parsed = tf.io.parse_single_example(record, features)
        # input_ids = tf.sparse.to_dense(parsed['input_ids'])
        # label_ids = tf.sparse.to_dense(parsed['label_ids'])
        input_ids = parsed['input_ids']
        label_ids = parsed['label_ids']
        return {'input_ids': input_ids}, label_ids

    if is_training:
        dataset = tf.data.TFRecordDataset(tfrecord_file).map(
            _parse_fn).shuffle(10000).batch(batch_size)
    else:
        dataset = tf.data.TFRecordDataset(tfrecord_file).map(_parse_fn).batch(
            batch_size)
    return dataset


def train():
    config = basic_config()
    print('检查数据是否写入tfrecord')
    to_tf_record(config)
    train_file = os.path.join(config.data_save_path, 'train.tfrecord')
    dev_file = os.path.join(config.data_save_path, 'dev.tfrecord')
    test_file = os.path.join(config.data_save_path, 'test.tfrecord')

    train_dataset = to_tf_dataset(tfrecord_file=train_file,
                                  batch_size=config.batch_size,
                                  is_training=True,
                                  max_length=config.max_length)

    dev_dataset = to_tf_dataset(tfrecord_file=dev_file,
                                batch_size=config.batch_size,
                                is_training=False,
                                max_length=config.max_length)

    test_dataset = to_tf_dataset(tfrecord_file=test_file,
                                 batch_size=config.batch_size,
                                 is_training=False,
                                 max_length=config.max_length)
    
    strategy = tf.distribute.MirroredStrategy()
    # 1. 定义数据集，分布式
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dev_dataset = strategy.experimental_distribute_dataset(dev_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)
    with strategy.scope():
        # 2. 定义loss函数
        loss_object = loss_fun

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=config.batch_size)

        # 3. 定义metrics
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = Accuracy(name='train_acc')
        test_accuracy = Accuracy(name='test_acc')

        model = TFGPT2LMHeadModel.from_pretrained(config.pretrained_model_path)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def train_step(inputs):
        train_inputs, labels = inputs
        with tf.GradientTape() as tape:
            predictions = model(train_inputs, training=True)
            loss = compute_loss(labels, predictions.logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions.logits)
        return loss

    def test_step(inputs):
        test_inputs, labels = inputs
        predictions = model(test_inputs, training=False)
        loss = loss_object(labels, predictions.logits)
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, predictions.logits)

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs, ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,
                               per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(test_step, args=(dataset_inputs, ))

    for n_epoch in range(config.epoch):
        print("epoch：{}".format(n_epoch + 1))
        total_loss = 0.0
        num_batches = 0
        for x in train_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1

            if num_batches > 0 and num_batches % 2 == 0:
                print("Epoch: {}, Loss: {}, Acc: {}".format(
                    n_epoch + 1, total_loss / num_batches,
                    train_accuracy.result() * 100))
        train_loss = total_loss / num_batches
        for x in test_dataset:
            distributed_test_step(x)
        print("Save ckpt")
        checkpoint_prefix = os.path.join(config.model_save_path, 'ckpt')
        if not os.path.isdir(checkpoint_prefix):
            os.makedirs(checkpoint_prefix)

        checkpoint.save(checkpoint_prefix)
        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print(
            template.format(n_epoch + 1, train_loss,
                            train_accuracy.result() * 100, test_loss.result(),
                            test_accuracy.result() * 100))
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
    for x in dev_dataset:
        distributed_test_step(x)
    print("dev, loss:{} acc:{}".format(test_loss.result(),
                                       test_accuracy.result() * 100))

class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.true = self.add_weight(name='true',
                                    initializer='zeros',
                                    dtype='float32')
        self.total = self.add_weight(name='total',
                                     initializer='zeros',
                                     dtype='float32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=-1), shape=(-1, ))
        y_true = tf.reshape(y_true, (-1, ))
        mask = tf.not_equal(y_true, -100)

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)

        mask = tf.cast(mask, 'float32')

        self.true.assign_add(tf.reduce_sum(values))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        # if tf.equal(self.total,0.0):
        #     return 0.0
        return self.true / self.total

    def reset_states(self):
        self.true.assign(0)
        self.total.assign(0)


def loss_fun(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    # make sure only labels that are not equal to -100 affect the loss
    active_loss = tf.not_equal(tf.reshape(labels, (-1, )), -100)
    reduced_logits = tf.boolean_mask(
        tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
    labels = tf.boolean_mask(tf.reshape(labels, (-1, )), active_loss)
    return loss_fn(labels, reduced_logits)


if __name__=="__main__":
    train()
