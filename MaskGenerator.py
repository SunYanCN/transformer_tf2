import tensorflow as tf


def create_padding_mask(seq):
    """ 生成mask，mask值为1 """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    # 执行attention计算时，attention_matrix=[batch_size, num_head, seq_len_q, seq_len_k]
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """ 生成上三角矩阵，对角线置0（可见），即不可见序列的右侧，可见当前输入和左侧 """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_mask(inp, tar):
    """
    创建mask
    :param inp: encoder的输入
    :param tar: decoder的输入, 首位是[start]的目标序列
    :return:
    """
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])  # 预测位置的右侧mask
    dec_target_padding_mask = create_padding_mask(tar)  # 超过序列长度mask
    combine_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # 逻辑或：当前位置超过序列长度，或在预测位置的右侧时，mask
    print(combine_mask)
    return enc_padding_mask, combine_mask, dec_padding_mask


if __name__ == '__main__':
    pass
