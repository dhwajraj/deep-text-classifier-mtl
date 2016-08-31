It is a tensorflow implementation using MULTI-TASK LEARNING for Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

Core methods are derived from [dennybritz implementation](https://github.com/dennybritz/cnn-text-classification-tf)
The major refactoring has been done to incorporate the following:
 - Loading pre-trained word embeddings
 - Loading tab separated training text (format : label<tab>text<newline>)
 - Training multiple different binary classification tasks at once (Multi-Task Learning)


![multi task learning](https://cloud.githubusercontent.com/assets/9861437/18117883/233370b8-6f6f-11e6-8409-15e7ca5a7541.png)

## Method
Following diagram is depicting the deep architecture for a single binary text classification task using Convolutional Neural Networks. Image taken from Ye Zhang's paper.
![deep text classifier CNN](https://cloud.githubusercontent.com/assets/9861437/18117883/233370b8-6f6f-11e6-8409-15e7ca5a7541.png)

# Why Multi-Task Learning

In multi-task alternative training, same model is alternatively trained to perform multiple binary classification tasks in the same language.
![multi task learning](https://cloud.githubusercontent.com/assets/9861437/18117883/233370b8-6f6f-11e6-8409-15e7ca5a7541.png)

Multi-task training can exploit the fact that different sequence tagging tasks in one language share language-specific regularities. The basic idea is to share part of the architecture and parameters between tasks, and to alternatively train multiple objective functions with respect to different tasks. Tensorflow automatically figures out which calculations are needed for the operation you requested, and only conducts those calculations. This means that if we define an optimiser on only one of the tasks, it will only train the parameters required to compute that task - and will leave the rest alone. Since Task 1 relies only on the Task 1 and Shared Layers, the Task 2 layer will be untouched. Let’s draw another diagram with the desired optimisers at the end of each task.



## Requirements

- Python 3
- Tensorflow > 0.8
- Numpy

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
./train.py
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
- [Jonathan Godwin's explanation of multi-task learning in Tensorflow](http://www.kdnuggets.com/2016/07/multi-task-learning-tensorflow-part-1.html)
