# SAFEPG
![img](https://release-badges-generator.vercel.app/api/releases.svg?user=YikaiZhang95&repo=SAFE&gradient=0000ff,8bd1fa)

A Novel SAFE Model for Predicting Climate-Related Extreme Losses

## Table of contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Usage](#usage)
* [Getting help](#getting-help)

## Introduction

The frequency-severity model has been widely adopted to analyze highly right-skewed data 
in actuarial science. To make the model more interpretable, we expect a predictor has 
the same direction of impact on both the frequency and severity. However, the 
compotemporary use of the frequence-severity model typically yields inconsistent signs. 
To this end, we propose a novel sign-aligned regularization term to facilitate the sign 
consistency between the components in the frequency-severity model to enhance interpretability. 
We also demonstrate our design of the penalty leads to an algorithm which is quite efficient 
in analyzing large-scale data and its superior performance with both simulation and real examples.


## Installation

You can use `pip` to install this package.

```sh
pip install SAFEPG
```


## Quick start

The usages are similar with `scikit-learn`:

```python
model = SafeModel()
model.fit(x=x, y=y, k=k, lambda_=ulam)
```

## Usage

### Generate simulation data
`SAFEPG` provides a simulation data generation function to test functions in the library:

```python
from SAFEPG.SAFEPG import SafeModel
import numpy as np
from scipy.stats import poisson, gamma

np.random.seed(0)
n = 100
p = 5
x = np.random.randn(n, p)
beta_true = np.full(5, 0.1)
gamma_true = np.array([1, 1, 1, -1, -1])

mu = x @ beta_true
k = poisson.rvs(mu=np.exp(mu))
alpha_val = 1
theta = np.exp(x @ gamma_true) / alpha_val
y = gamma.rvs(a=alpha_val, scale=theta)

lambda_val = [1.0]
ind_p = np.array([1, 1, 1, 0, 0])

model = SafeModel()
model.fit(x=x, y=y, k=k, lambda_=lambda_val, ind_p = ind_p)
```

## Getting help
Any questions or suggestions please contact: <yikai-zhang@uiowa.edu>

