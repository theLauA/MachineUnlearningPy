# Introduction

This is an implementation of the unlearning algorithms presented in Toward Making Systems Forget with Machine Learning, written by Cao, Yinzhi and Yang, Junfeng. Specifically, we are implementing the unlearning algorithm for LensKit's ItemItem similarity algorithm.

# SetUp

* Python 3 is required
* Download the MovesLens Dataset from https://grouplens.org/datasets/movielens/100k/
    * Unzip and save `./ml-100k` at `./`
* Run `./unlearn/basic.py` to start the pipline of time cost evaluations.

# Modification

Modifications are made in `./lenskit/algorithms/item_knn.py`  from line `221` to `691`. \
Codes are injected in `./lenskit/algorithms/item_knn.py` method `fit` to run time cost evaluation and save results in `.csv`\
A pipeline is written in `./unlearn/basic.py` to run produce time cost evaluation for different input size \
A visualization is written in `./unlearn/visualization.py` to graph the time cost evaluation stored in `.csv`



# Python recommendation tools

[![Build Status](https://dev.azure.com/md0553/md/_apis/build/status/lenskit.lkpy)](https://dev.azure.com/md0553/md/_build/latest?definitionId=1)
[![codecov](https://codecov.io/gh/lenskit/lkpy/branch/master/graph/badge.svg)](https://codecov.io/gh/lenskit/lkpy)
[![Maintainability](https://api.codeclimate.com/v1/badges/c02098c161112e19c148/maintainability)](https://codeclimate.com/github/lenskit/lkpy/maintainability)

LensKit is a set of Python tools for experimenting with and studying recommender
systems.  It provides support for training, running, and evaluating recommender
algorithms in a flexible fashion suitable for research and education.

Python LensKit (LKPY) is the successor to the Java-based LensKit project.
