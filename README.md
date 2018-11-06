# Bayesian Optimization of Pointnet
Hyperparameter optimization using bayesian optimization on the popular Pointnet framework based on the works of [Revisiting Small Batch Training for Deep Neural Networks by Dominic Masters, Carlo Luschi](https://arxiv.org/abs/1804.07612) and [Algorithms for hyper-parameter optimization by James Bergstra, R´emi Bardenet, Yoshua Bengio, Bal´azs K´egl] (https://dl.acm.org/citation.cfm?id=2986743)

## Prerequisites
* Git
* Anaconda 1.9.2
* Jupyter Notebook 5.5.0
* Spyder 4.0
* Python 3.6

## Libraries
* hyperopt
* tensorflow
* tensorboard  


## Setup
Copy Scripts/skfuzzy/controlsystem.py to <Anaconda Installation Folder>\Lib\site-packages\skfuzzy\control. This is required to use the get antecedent function custom implementation.

## Running the solution
Please run the following commands on a terminal.
* git clone https://github.com/steve7an/PointnetEnhanced.git
* jupyter notebook
* Open Pointnet_Training_and_Evaluate.ipynb on notebook
* run each section independently to test Pointnet, Pointnet++ or 3DmFV

Hyperopt is currently only enabled for Pointnet.
