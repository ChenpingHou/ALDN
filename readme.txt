# # Code of ALDN(**Adaptive Learning for Dynamic Features and Noisy Labels**)

## Overview
This is the code for the paper "Adaptive Learning for Dynamic Features and Noisy Labels", which proposes a novel adaptive learning method for dynamic features and noisy labels (ALDN). There are two main stages in the algorithm:
1. Data in P-stage contains clean labeled data, and we first use data in P-stage to build  a well learned model W1.
2. Data in C-stage contains noisily labeled data with limited  samples and different features. We reuse the model W1 learned in P-stage to train a new model on data in C-stage.

## Main Files
There are two main files to implement ALDN method in the paper. 
 - Ours_reg1.m: the method using direct constraint. There are five input parameters. 
                - vanish_rate: the rate of vanished features.
                - noise_M: The estimated noise transition matrix
                - W1: The model learned in P-stage
                - T: The optimal transport matrix
 	- data_s2: the current training data. Each column represents an instance.
                - label_s2: the true label of current training data. Each column represents a label.
                - noise_label_s2: the noisy label of current training data. Each column represents a label.
 	- alpha_set: the parameter set.
 - Ours_reg1.m: the method using indirect constraint. There are five input parameters. 
                - vanish_rate: the rate of vanished features.
                - noise_M: The estimated noise transition matrix
                - W1: The model learned in P-stage
                - T: The optimal transport matrix
 	- data_s2: the current training data. Each column represents an instance.
                - label_s2: the true label of current training data. Each column represents a label.
                - noise_label_s2: the noisy label of current training data. Each column represents a label.
 	- alpha_set: the parameter set.

- demo.m: script that you can directly run on synthetic dataset. We use mysoftmax.m to build W1 in P-stage.

We have uploaded the code and data to the web disk, the URL and extraction code are as follows:
URL：https://pan.baidu.com/s/1KPn7zK6NxA7kiDTdw1lTxg 
Extraction Code：0627

## Contact:
If there are any problems, please feel free to contact Chenping Hou (hcpnudt@hotmail.com).