# 03 Deep Learning Basics

We're going to train a neural network to do a XOR operation. XOR is non-linear operation. Approximating non-linear functions and dealing with high dimensional inputs is the goal of neural networks. Here we take a look at non-linearity.

A B OUT
0 0 0
0 1 1
1 1 0
1 0 1

Deep learning has a set of building blocks:
- [x] neural networks. most simple neural net is a multi-layer perceptron. we need those to be differentiable.
- [x] loss functions: used to the evaluate the output of a neural net vs, the desired output
- [x] autograd: machinery to compute the gradients of the parameters of the neural net w.r.t the loss. implements backpropagation.
- [x] gradients: how to change the parameters such that they minimize the loss 
- [x] batch: we want take advantage of parallel processors (mainly GPUs) such that we train on a lot of data
  to do so, we batch our inputs/outputs

homework:
- either watch Karpathy's video on backprop: https://www.youtube.com/watch?v=VMj-3S1tku0
- or, reimplement the xor neural net experiment
