# The MIT License
#
# Copyright (c) VD44
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""ROUGE L metric implementation using tensorflow ops.


Some methods extended from:
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py.
"""

import tensorflow as tf

def tf_rouge_l(hyp_t, ref_t, end_id):
    """
    Computes ROUGE-L (sentence level) of two text collections of sentences (as Tensors).
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    Calculated according to:
    R_lcs = LCS(X,Y)/m
    P_lcs = LCS(X,Y)/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    X = candidate summary
    Y = reference summary
    m = length of candidate summary
    n = length of reference summary
  
    Args:
      hyp_t: Tensor containing sequences to be evaluated
      ref_t: Tensor containing references
      end_id: end token, used to truncate sequences before calculating ROUGE-L
    
    Returns:
      1-D Tensor. Average ROUGE-L values
    """
    # expand first dim of sequences if rank is 1
    hyp_t = tf.cond(tf.equal(tf.rank(hyp_t), 1), lambda: tf.expand_dims(hyp_t, 0), lambda: hyp_t)
    ref_t = tf.cond(tf.equal(tf.rank(ref_t), 1), lambda: tf.expand_dims(ref_t, 0), lambda: ref_t)
    batch_size = tf.shape(hyp_t)[0]
    # get indexes of end tokens in each pair of sentences
    m = tf.cast(tf.argmax(tf.cast(tf.equal(hyp_t, end_id), tf.float32), 1), tf.int32)
    n = tf.cast(tf.argmax(tf.cast(tf.equal(ref_t, end_id), tf.float32), 1), tf.int32)
    # don't truncate sequences if no end token
    m = tf.where(tf.equal(m,0),tf.fill([batch_size], tf.shape(hyp_t)[1]-1),m)
    n = tf.where(tf.equal(n,0),tf.fill([batch_size], tf.shape(ref_t)[1]-1),n)
    
    k = 0
    total_rouge = tf.constant([0., 0., 0.])
    def _step(k, total_rouge):
        # calculate ROUGE-L for every element
        llcs = _tf_len_lcs(hyp_t[k], ref_t[k], m[k], n[k])
        res = _f_p_r_lcs(llcs, m[k], n[k])
        res = tf.cast(res, tf.float32)
        return k + 1, total_rouge + res
    # run loop
    _, total_rouge = tf.while_loop(
    cond=lambda k, *_: k<batch_size,
    body=_step,
    loop_vars=[k, total_rouge])
    # get average ROUGE-L values
    rouge = total_rouge / tf.cast(batch_size, tf.float32)
    return rouge

def _f_p_r_lcs(llcs, m, n):
    """
    Computes the LCS-based F-measure score.
    Source: https://www.microsoft.com/en-us/research/publication/
    researchouge-a-package-for-automatic-evaluation-of-summaries/
  
    Args:
      llcs: Length of LCS
      m: number of words in (truncated) candidate summary
      n: number of words in (truncated) reference summary
    
    Returns:
      Float. LCS-based F-measure score
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta**2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta**2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs, p_lcs, r_lcs

def _tf_len_lcs(x, y, m, n):
    """
    Computes the length of the LCS between two truncated seqs.

    Args:
      x: first seq
      y: second seq
      m: length to which x is truncated
      n: length to which y is truncated

    Returns:
      0-D Tensor. Length of LCS 
    """
    table = _tf_lcs(x, y, m, n)
    return table[m, n]

def _tf_lcs(x, y, m, n):
    """
    Computes the LCS between two truncated seqs.
    The implementation below uses a DP programming algorithm and runs
    in O(mn) time where m = len(x) and n = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: first seq
      y: second seq
      m: length to which x is truncated (values in x after length m are disregarded)
      n: length to which x is truncated (values in y after length m are disregarded)

    Returns:
      2-D Tensor. Table used to find length of LCS
    """
    table_size = (m+1)*(n+1)
    k = 0
    table = tf.TensorArray(tf.int32, table_size, clear_after_read=False, element_shape=[])
    def loop_step(k, table):
        # get col and row index at current step (table is a 1d array but we treat it as a 2d array)
        j = tf.cast(k % (n+1), tf.int32)
        i = tf.cast((k - j) / (n+1), tf.int32)
        # get and write value for current index
        val = tf.cond(tf.logical_or(tf.equal(i,0),tf.equal(j,0)),
            lambda:0,
            lambda:tf.cond(tf.equal(x[i-1],y[j-1]),
            lambda:table.read((i-1)*(n+1)+j-1)+1,
            lambda:tf.maximum(table.read((i-1)*(n+1)+j),table.read(i*(n+1)+j-1))))
        table=table.write(k, val)
        return k+1,table
    # run loop
    _, table = tf.while_loop(
    cond=lambda k, *_: k<table_size,
    body=loop_step,
    loop_vars=[k, table])
    # stack and reshape table to 2d shape
    table = tf.reshape(table.stack(), [m+1,n+1])
    return table
