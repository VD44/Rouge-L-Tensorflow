# Rouge-L-Tensorflow
ROUGE L metric implementation using tensorflow ops.

Implemented for the purpose of reinforcing a summary model without having to switch to CPU for computation.

## About
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is specified in the paper [ROUGE: A Package for Automatic Evaluation of Summaries](http://www.aclweb.org/anthology/W04-1013). It is often used as a method to automatically evaluate machine generated summeries.

Can be used to compute Rouge L during training as a reinforment objective as described in the paper [A Deep Reinforced Model For Abstractive Summarization](https://arxiv.org/pdf/1705.04304.pdf), without switching to CPU. Significantly more efficient than using tf.py_func with a python Rouge L function.

## Usage
Function tf_rouge_l takes three arguments: a Tensor to evaluate, a Tensor to act as a reference, and an end id to truncate each sequence with. If the end id is not found within a sequence, the sequence is not truncated.

```python
import tensorflow as tf
from rouge_l_tensorflow import tf_rouge_l

sess = tf.Session()

val = tf.constant([[5,6,13,57,100,4,5,0,0], [67,55,31,52,88,100,0,0,0]])
ref = tf.constant([[5,6,13,57,100,4,0,9,88], [67,99,31,57,88,4,0,0,4]])
end_token = 100

sess.run(tf_rouge_l(val, ref, end_token)) 
# => array([ 0.70957613,  0.6875,  0.80000001], dtype=float32)
```
