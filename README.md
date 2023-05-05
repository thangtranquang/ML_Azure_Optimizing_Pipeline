# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

This dataset contains data about individuals applying for bank loans. The task we set out to accomplish here is to develop a model that,
based on the information provided about each individual, predicts whether they will subscribe to a service.

The best performing model was 90.2% accuracy with LogisticRegression

## Scikit-learn Pipeline

Step the pipeline:

1. Downloading dataset
2. Clean data:
- Removing NAs
- One-hot encoding: jobs, contact and education
- Encoding a number:marital, default, housing, loan, poutcome
- Encoding months and weekdays of the year.
- Encoding the target

3. Split into a training and test set with 80% training and 20% testing  to reduce train error andthe gap between train error and test error.
4. Model used: LogisticRegression with parameters: --C help Inverse of regularization strength, and --max_iter help Maximum number of iterations to converge
Once the data has been prepared it is split into a training and test set. A test set size of 33% of total entries was selected as a compromise between ensuring adequate representation in the test data and providing sufficient data for model training. 
5. Early stopping policy: MedianStoppingPolicy to avoid overfitting when training a learner

## AutoML

The autoML pipeline:
- Load data
- Clean data
- Split into train and test
- Model config: LogisticRegression with parameters: --C, --max_iter 

The result:  90.2% accuracy

## Pipeline comparison
**Compare the two models and their performance:

- VotingEnsemble: 0.9170 using AutoML model
- LogisticRegression: 0.902

Conclusion: With AutoML model is actually better because it make all the necessary calculations, trainings, validations.

## Future work

This data is highly imbalanced. Class imbalance is a very common issue in classification problems in machine learning.

To deal with imbalanced data:
- Pre-processing data: Random Under-Sampling of majority class, Random Over-Sampling of minority class
- Use different metric as F1 score
- Use different algorithms

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
- as image in github
