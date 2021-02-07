import tensorflow as tf

class WordACC(tf.keras.metrics.Metric):
    """
    Calculate the word accuracy between y_true and y_pred.
    """
    def __init__(self, name='word_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32,
                                     initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32,
                                     initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Maybe have more fast implementation.
        """
        size = tf.shape(y_true)[0]
        max_width = tf.maximum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        # 0： height (size of 2-dim array)
        # 1: row (rows of each 2-dim array)
        # 2: column (columns of each 2-dim array)
        # perm = [1, 0, 2] 转置了 height 和 row
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        y_true = tf.sparse.reset_shape(sp_input=y_true, new_shape=[size, max_width])
        y_pred = tf.sparse.reset_shape(sp_input=decoded[0], new_shape=[size, max_width])
        y_true = tf.sparse.to_dense(y_true, default_value=-1)
        y_pred = tf.sparse.to_dense(y_pred, default_value=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        values = tf.math.reduce_any(tf.math.not_equal(y_true, y_pred), axis=1)
        values = tf.cast(values, tf.int32)
        values = tf.reduce_sum(values)

        self.total.assign_add(size)
        self.count.assign_add(size - values)

    def result(self):
        print(self.count, '/', self.total)
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)