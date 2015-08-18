# Recurrency

A framework for experimenting with Recurrent Neural Network architectures.

Maintainer - [Pasquale Minervini](http://github.com/pminervini)

## Sample Usage

Toy task - memorize the first element of a sequence of 30 binary elements, training a simple RNN with 10 hidden units using AdaGrad:


```
recurrency$ PYTHONPATH='.' ./examples/first.py --rnn --optimizer=adagrad --rate=0.1 --length=30 --hidden=10
INFO:root:[RNN(1, 10, 1) 0]	2016.70841816
INFO:root:[RNN(1, 10, 1) 1]	510.591709843
INFO:root:[RNN(1, 10, 1) 2]	480.646542102
INFO:root:[RNN(1, 10, 1) 3]	410.613361412
INFO:root:[RNN(1, 10, 1) 4]	360.961450817
INFO:root:[RNN(1, 10, 1) 5]	311.325939509
INFO:root:[RNN(1, 10, 1) 6]	269.729312314
INFO:root:[RNN(1, 10, 1) 7]	228.864181037
INFO:root:[RNN(1, 10, 1) 8]	187.205097576
INFO:root:[RNN(1, 10, 1) 9]	145.889039444
INFO:root:[RNN(1, 10, 1) 10]	115.774753624
INFO:root:[RNN(1, 10, 1) 11]	96.9229396861
INFO:root:[RNN(1, 10, 1) 12]	78.2241894971
INFO:root:[RNN(1, 10, 1) 13]	59.8859616697
INFO:root:[RNN(1, 10, 1) 14]	48.3193670917
```


## References

Recurrent Neural Networks, (Projected) Long Short-Term Mememory Networks:

- Martens, J. et al. - [Learning Recurrent Neural Networks with Hessian-Free Optimization](http://www.icml-2011.org/papers/532_icmlpaper.pdf) - ICML 2011
- Sak, H. et al. - [Long Short-Term Memory Recurrent Neural Network Architectures
for Large Scale Acoustic Modeling](https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenTerm1201415/sak2.pdf) - INTERSPEECH 2014
- Sak, H. et al. - [Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech Recognition](http://arxiv.org/abs/1402.1128) - 	arXiv:1402.1128

Gated Recurrent Units, Gated-Feedback Recurrent Neural Networks:

- Chung, J. et al. - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555) - arXiv:1412.3555
- Chung, J. et al. - [Gated Feedback Recurrent Neural Networks](http://arxiv.org/abs/1502.02367) - arXiv:1502.02367

Sequence-to-Sequence Learning (e.g. for Machine Translation):

- Sutskever, I. et al. - [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - NIPS 2014


Parameters initialization and learning:

- Glorot, X. et al. - [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) - AISTATS 2010
- Martens, J. et al. - [Learning Recurrent Neural Networks with Hessian-Free Optimization](http://www.icml-2011.org/papers/532_icmlpaper.pdf) - ICML 2011
- Le, Quoc V. et al. - [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units](http://arxiv.org/abs/1504.00941) - arXiv:1504.00941
