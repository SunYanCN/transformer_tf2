import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class PositionalEncoder:
    def __init__(self, d_model: int, position: int):
        self.d_model = d_model
        self.position = position

    @staticmethod
    def get_angels(pos, i, d_model):
        '''
        :param pos:
        :param i:
        :param d_model: model dimension
        :return:
        (2*(1//2)) 保证是2的整数倍
        angle_rates = pos ( 1   /   [  1000 ** (  (2*(i//2)) / d_model   )  ) ]
        '''
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self):

        angle_rads = self.get_angels(np.arange(self.position)[:, np.newaxis],
                                     np.arange(self.d_model)[np.newaxis, :],
                                     self.d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        # official version
        # pos_encoding = np.concatenate([sines, cosines], axis=-1)
        # pos_encoding = pos_encoding[np.newaxis, ...]

        # less memo version
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype = tf.float32)


if __name__ == '__main__':
    # ************************************
    # 测试position encoder
    # ************************************
    pos_encoding = PositionalEncoder(50, 512).positional_encoding()
    print(pos_encoding.shape)
    plt.pcolormesh(pos_encoding[0], cmap = 'RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
