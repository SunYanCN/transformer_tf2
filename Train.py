import os
import sys

sys.path.append("../")
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # 单GPU的机器，0是GPU，1是CPU
import time
import datetime

from tensorflow import keras
from MaskGenerator import *

from Transformer import Transformer
from DataLoader import gen_dataset
from cfg import *


# 自定义lr优化
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# 损失函数为rmse
def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype = loss_.dtype)
    return tf.reduce_mean(loss_ + (mask * -1e9))


# model evaluation
def validate(transformer, dataset, eval_type = 'Val'):
    """ 验证或测试 """
    print("*** Running {} step ***".format(eval_type))
    accuracy = keras.metrics.SparseCategoricalAccuracy(name = '{}_accuracy'.format(eval_type.lower()))

    for (batch, (inp, tar)) in enumerate(dataset):
        # 对于seq来说，对于seq[n], 其input为seq[:n-1]
        # 因此对于所有input,其target为seq[1:]
        # tar_input = tar[:, :-1]，首位是[start]的目标序列
        # tar_real = tar[:, 1:] 目标序列本身

        tar_input = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combine_mask, dec_padding_mask = create_mask(inp, tar_input)
        predictions, _ = transformer(inp,
                                     tar = tar_input,
                                     training = False,
                                     enc_padding_mask = enc_padding_mask,
                                     look_ahead_mask = combine_mask,
                                     dec_padding_mask = dec_padding_mask)

        accuracy(tar_real, predictions)

        if batch % 100 == 0:
            print('Batch {} Accuracy {:.4f}'.format(batch, accuracy.result()))

    print('\n{} Accuracy {}\n'.format(eval_type, accuracy.result()))


def train():
    """ 训练入口 """
    # ************************************
    # 1. 准备数据集
    # ************************************
    train_dataset, val_dataset, input_vocab_size, target_vocab_size = gen_dataset(CONFIG['model']['batch_size'])

    # ************************************
    # 2. 机器学习三大组件：模型，损失函数，优化器
    # ************************************
    d_model = CONFIG['model']['d_model']
    learning_rate = CustomSchedule(d_model)
    optimizer = keras.optimizers.Adam(learning_rate, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9)
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

    train_loss = keras.metrics.Mean(name = 'train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

    transformer = Transformer(
        num_layers = CONFIG['model']['num_layers'],
        d_model = CONFIG['model']['d_model'],
        num_heads = CONFIG['model']['num_heads'],
        dff = CONFIG['model']['dff'],
        input_vocab_size = input_vocab_size,
        target_vocab_size = target_vocab_size
    )

    # ************************************
    # 3. 训练流程的准备
    # ************************************
    # 定义checkpoints管理器
    checkpoint_path = get_path(CONFIG['checkpoint_path'])
    ckpt = tf.train.Checkpoint(model = transformer, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = 5)

    # 使用Tensorboard可视化训练过程的loss和accuracy
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = get_path('logs/gradient_tape/' + current_time + '/train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # @tf.function
    def train_step(inp, tar):
        tar_input = tar[:, :-1]  # 目标序列首位插入了开始符 [start]，相当于做了右移
        tar_real = tar[:, 1:]  # 目标序列本身

        enc_padding_mask, combine_mask, dec_padding_mask = create_mask(inp, tar_input)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp,
                                         tar = tar_input,
                                         training = True,
                                         enc_padding_mask = enc_padding_mask,
                                         look_ahead_mask = combine_mask,
                                         dec_padding_mask = dec_padding_mask)
            loss = loss_function(tar_real, predictions, loss_object)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    # ************************************
    # 4. 开始训练
    # ************************************
    for epoch in range(CONFIG['model']['epochs']):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            # 记录loss和accuracy
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step = epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step = epoch)

            if batch % 100 == 0:
                print('Epochs {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        # 执行验证step
        validate(transformer, val_dataset)


if __name__ == '__main__':
    # ************************************
    # 测试 padding_mask
    # ************************************
    # seq = tf.constant([[1.0, 2.0, 0], [3.0, 4.0, 0]])
    # padding_mask = create_padding_mask(seq)
    # print(padding_mask.numpy())
    # ************************************
    # 测试 create_look_ahead_mask
    # ************************************
    # look_ahead_mask = create_look_ahead_mask(15)
    # print(look_ahead_mask.numpy())

    # ************************************
    # 测试 CustomSchedule
    # ************************************
    # import matplotlib.pyplot as plt
    # temp_learning_rate_schedule = CustomSchedule(128)
    # plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.show()\
    train()
