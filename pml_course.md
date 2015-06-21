---
title: "Human Activity Recognition"
output: html_document
author: Dmytro Kliagin
---

# Overview

In this document we attempt to fit a model to predict the type of a physical activity
performed by people based on the data from wearable accelerometers.

Detailed information on the dataset is available
[http://groupware.les.inf.puc-rio.br/har](through this link).

The dataset contains wearables observation points for six different people
at different time. From the description of the dataset:

    Six young health participants were asked to perform one set of 10 repetitions
    of the Unilateral Dumbbell Biceps Curl in five different fashions:
    exactly according to the specification (Class A),
    throwing the elbows to the front (Class B),
    lifting the dumbbell only halfway (Class C),
    lowering the dumbbell only halfway (Class D)
    and throwing the hips to the front (Class E).

# Fitting the Model

Required libraries:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(doParallel)
```

```
## Loading required package: foreach
## foreach: simple, scalable parallel programming from Revolution Analytics
## Use Revolution R for scalability, fault tolerance and more.
## http://www.revolutionanalytics.com
## Loading required package: iterators
## Loading required package: parallel
```

As the first step, we're reading the data. It is assumed the files are already downloaded.
Looking at the downloaded files, there are a number of values that are empty or NA, or "#DIV/0!",
so we have to clean data out of those:


```r
train_data <- read.csv("pml-training.csv", header=TRUE, quote="\"", na.strings = c("NA", "", "#DIV/0!"))
test_data <- read.csv("pml-testing.csv", header=TRUE, quote="\"", na.strings = c("NA", "", "#DIV/0!"))
```

Input files contain observations for a large number of variables.
A lot of them may not be relevant to fitting the model.
For example, most people do the same excercise in roughly similar way regardless from when they
are doing it, so we can exclude this variables:


```r
irrelevant <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
```

Large number of variables have mostly NA values and are somehow derived from other data, for example,
max/min value, variance, standard deviation, etc. Those variables should be excluded as well:


```r
cols <- colnames(train_data)
irrelevant <- c(irrelevant, cols[grepl("^max", cols)])
irrelevant <- c(irrelevant, cols[grepl("^min", cols)])
irrelevant <- c(irrelevant, cols[grepl("^stddev", cols)])
irrelevant <- c(irrelevant, cols[grepl("^var", cols)])
irrelevant <- c(irrelevant, cols[grepl("^avg", cols)])
irrelevant <- c(irrelevant, cols[grepl("^total", cols)])
irrelevant <- c(irrelevant, cols[grepl("^amplitude", cols)])
irrelevant <- c(irrelevant, cols[grepl("^kurtosis", cols)])
irrelevant <- c(irrelevant, cols[grepl("^skewness", cols)])
```

In order to estimate out of sample error, let's split our train_data into parts to fit a model
and to do cross-validation:


```r
set.seed(123)
inTrain <- createDataPartition(y = train_data$classe, p = 0.6, list = FALSE)
training <- train_data[inTrain, sapply(cols, function(c) { !(c %in% irrelevant) } )]
crossval <- train_data[-inTrain, sapply(cols, function(c) { !(c %in% irrelevant) } )]
```

Let's fit boosted tree model:


```r
cluster <- makeCluster(detectCores())
registerDoParallel(cluster)
modelFit <- train(classe ~ ., data = training, method = "gbm")
```

```
## Loading required package: gbm
```

```
## Warning: package 'gbm' was built under R version 3.1.3
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loaded gbm 2.1.1
## Loading required package: plyr
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2346
##      2        1.4602             nan     0.1000    0.1609
##      3        1.3574             nan     0.1000    0.1239
##      4        1.2795             nan     0.1000    0.1026
##      5        1.2147             nan     0.1000    0.0908
##      6        1.1576             nan     0.1000    0.0779
##      7        1.1081             nan     0.1000    0.0756
##      8        1.0603             nan     0.1000    0.0575
##      9        1.0229             nan     0.1000    0.0553
##     10        0.9878             nan     0.1000    0.0482
##     20        0.7561             nan     0.1000    0.0286
##     40        0.5314             nan     0.1000    0.0104
##     60        0.4035             nan     0.1000    0.0070
##     80        0.3229             nan     0.1000    0.0044
##    100        0.2641             nan     0.1000    0.0042
##    120        0.2218             nan     0.1000    0.0016
##    140        0.1888             nan     0.1000    0.0013
##    150        0.1747             nan     0.1000    0.0015
```

Estimating out of sample error:


```r
predictions <- predict(modelFit, newdata = crossval)
confusionMatrix(predictions, crossval$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2195   32    0    1    5
##          B   21 1443   59    5   14
##          C    8   36 1292   40   13
##          D    4    4   12 1232   23
##          E    4    3    5    8 1387
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9621          
##                  95% CI : (0.9577, 0.9663)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9521          
##  Mcnemar's Test P-Value : 1.836e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9834   0.9506   0.9444   0.9580   0.9619
## Specificity            0.9932   0.9844   0.9850   0.9934   0.9969
## Pos Pred Value         0.9830   0.9358   0.9302   0.9663   0.9858
## Neg Pred Value         0.9934   0.9881   0.9882   0.9918   0.9915
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2798   0.1839   0.1647   0.1570   0.1768
## Detection Prevalence   0.2846   0.1965   0.1770   0.1625   0.1793
## Balanced Accuracy      0.9883   0.9675   0.9647   0.9757   0.9794
```

# Summary

We fitted a tree model for the train dataset and received estimated 96% out of sample accuracy.
