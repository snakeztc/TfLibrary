# @Time    : 12/13/16 12:17 PM
# @Author  : Tiancheng Zhao
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def get_bleu_stats(ref, hyps):
    scores = []
    for hyp in hyps:
        try:
            scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1./3, 1./3,1./3]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld


def norm_log_liklihood(x, mu, logvar):
    return -0.5*tf.reduce_sum(tf.log(2*np.pi) + logvar + tf.div(tf.pow((x-mu), 2), tf.exp(logvar)), reduction_indices=1)


def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z


def sample_gumbel(shape, eps=1e-20):
    """Sample from gumbel(0,1) """
    u = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/temperature)


def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


def last_relevant(output, length, out_size, max_length):
    batch_size = tf.shape(output)[0]
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    relevant = tf.reshape(relevant, [batch_size, out_size])
    return relevant


def get_bow(embedding, avg=False):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    """
    embedding_size = embedding.get_shape()[2].value
    if avg:
        return tf.reduce_mean(embedding, reduction_indices=[1]), embedding_size
    else:
        return tf.reduce_sum(embedding, reduction_indices=[1]), embedding_size


def get_rnn_encode(embedding, cell, length_mask=None, scope=None, reuse=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    with tf.variable_scope(scope, 'RnnEncoding', reuse=reuse):
        if length_mask is None:
            length_mask = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(embedding), reduction_indices=2)),reduction_indices=1)
            length_mask = tf.to_int32(length_mask)
        _, encoded_input = tf.nn.dynamic_rnn(cell, embedding, sequence_length=length_mask, dtype=tf.float32)
        return encoded_input, cell.state_size


def get_bi_rnn_encode(embedding, f_cell, b_cell, length_mask=None, scope=None, reuse=None):
    """
    Assumption, the last dimension is the embedding
    The second last dimension is the sentence length. The rank must be 3
    The padding should have zero
    """
    with tf.variable_scope(scope, 'RnnEncoding', reuse=reuse):
        if length_mask is None:
            length_mask = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(embedding), reduction_indices=2)),reduction_indices=1)
            length_mask = tf.to_int32(length_mask)
        _, encoded_input = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, embedding, sequence_length=length_mask, dtype=tf.float32)
        encoded_input = tf.concat(encoded_input, 1)
        return encoded_input, f_cell.state_size+b_cell.state_size


def get_cnn_encode(embedding, filter_sizes, num_filters, max_utt_size, keep_prob, scope=None, reuse=None):
    with tf.variable_scope(scope, "CnnEncoding", reuse=reuse):
        embedding_size = embedding.get_shape()[2]
        expanded_embedding = tf.expand_dims(embedding, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable("W", shape=filter_shape, dtype=tf.float32)
                b = tf.get_variable("b", shape=[num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(
                    expanded_embedding,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_utt_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        if keep_prob < 1.0:
            h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

        return h_pool_flat, num_filters_total

