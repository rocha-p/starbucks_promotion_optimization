# Starbucks promotion campaign optimization.

In this project we test a promotion campagin made by Starbucks. The object is to perform an A/B test to evaluate the promotion results and then build a ML model in order to improve the campaign result choosing the better clients to target. 

Our promotion strategy will be evaluated on 2 key metrics
* Incremental Response Rate (IRR)
* Net Incremental Revenue (NIR)

## Table of Contents

1. [Installation and Libraries](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation and Libraries  <a name="installation"></a>

Python version: `3.8.5`

You will need to install the following libraries:

from itertools import combinations
from test_results import test_results, score
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import scipy.stats as stats
import matplotlib.pyplot as plt


## Project Motivation<a name="motivation"></a>
This project was offered by Starbucks in the context of the Udacity Data Science Nanodegree. 

The goal of the project is to evaluate a Starbucks promotion campaign based on two variable metrics using statistical analysis. 

The second part of the project builds a ML model to improve their promotion strategy.

## File Descriptions <a name="files"></a>
Repository structure:
    
    starbucks_promotion_optimization/        # folder
    ├── README.md                            # read.me file 
    ├── Test.csv                             # Data set used to evaluate results
    ├── training.csv                         # Data set - .csv file
    ├── Starbucks.ipynb                      # Project Notebook. Analysis and ML model
    ├── test_results.py                      # Python file provide by Udacity to evaluate results

## Results<a name="results"></a>

First we evaluate the invariant metric (Number of participants in each group) as a prerequisite so that the following inferences on the evaluation metrics are founded on solid ground. 

We fail to reject the $H_0$, so we can feel confident about go further with the variable metrics analysis.

Here we analyze two metrics: $IRR$ and $NIR$

The observed data about promotion campaign shows significant results for IRR metric but no good results in NIR.

We raised the idea that NIR could be improved using a machine learning model that choose the best customers to target the promotion, it means the customers with a higher probability of response.

We try two models: Random Forest and Easy Ensemble.

The two models improve the promotion performance.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

All the data was provided by Starbucks.

Use my code as you like. Any feedback is welcome!

