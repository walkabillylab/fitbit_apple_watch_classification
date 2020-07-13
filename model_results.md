---
title: "Using machine learning methods to predict physical activity types with Apple Watch and Fitbit data using indirect calorimetry as the criterion."
author: "Arastoo Bozorgi and Daniel Fuller"
date: "07/06/2019"
output:
      html_document:
        keep_md: true
---



## Loading the required libraries

```r
library("data.table")
library("dplyr")
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:data.table':
## 
##     between, first, last
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library("tidyverse")
```

```
## ── Attaching packages ─────────────────────────────────────────────────────────────────────────────── tidyverse 1.3.0 ──
```

```
## ✓ ggplot2 3.3.2     ✓ purrr   0.3.4
## ✓ tibble  3.0.2     ✓ stringr 1.4.0
## ✓ tidyr   1.1.0     ✓ forcats 0.5.0
## ✓ readr   1.3.1
```

```
## ── Conflicts ────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
## x dplyr::between()   masks data.table::between()
## x dplyr::filter()    masks stats::filter()
## x dplyr::first()     masks data.table::first()
## x dplyr::lag()       masks stats::lag()
## x dplyr::last()      masks data.table::last()
## x purrr::transpose() masks data.table::transpose()
```

```r
library("e1071")
library("caret")
```

```
## Loading required package: lattice
```

```
## 
## Attaching package: 'caret'
```

```
## The following object is masked from 'package:purrr':
## 
##     lift
```

```r
library("rpart")
library("rpart.plot")
library("mlbench")
library("randomForest")
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
library("stats")
#library("RWeka")
```

## Reading the data


```r
aggregated_data <- read_csv("aggregated_fitbit_applewatch_jaeger.csv")
```

```
## Parsed with column specification:
## cols(
##   .default = col_double(),
##   DateTime = col_datetime(format = ""),
##   Applewatch.Username = col_character(),
##   Applewatch.DeviceName = col_character(),
##   Fitbit.Username = col_character(),
##   Fitbit.DeviceName = col_character(),
##   activity = col_character(),
##   activity_trimmed = col_character(),
##   gender = col_character(),
##   time = col_datetime(format = ""),
##   TodurLockeLabels = col_character(),
##   TodurLockeLabels_LE = col_character()
## )
```

```
## See spec(...) for full column specifications.
```

### Descriptive statistics

```r
gender_stats <- aggregated_data %>% 
  group_by(id, gender) %>% 
  summarise(
     n()
  ) %>% 
  group_by(gender) %>% 
  summarise(
    freq = n()
    ) %>% 
  mutate(
    percent = (freq / sum(freq)) * 100
    )
```

```
## `summarise()` regrouping output by 'id' (override with `.groups` argument)
```

```
## `summarise()` ungrouping output (override with `.groups` argument)
```

```r
identity_stats <- aggregated_data %>% 
  group_by(gender) %>% 
  summarise(
    mean_age = mean(age),
    sd_age = sd(age),
    mean_height = mean(height),
    sd_height = sd(height),
    mean_weight = mean(weight),
    sd_weight = sd(weight),
    mean_Applewatch.Steps_LE = mean(Applewatch.Steps_LE),
    sd_Applewatch.Steps_LE = sd(Applewatch.Steps_LE),
    mean_Applewatch.Heart_LE = mean(Applewatch.Heart_LE),
    sd_Applewatch.Heart_LE = sd(Applewatch.Heart_LE),
    mean_Applewatch.Calories_LE = mean(Applewatch.Calories_LE),
    sd_Applewatch.Calories_LE = sd(Applewatch.Calories_LE),
    mean_Applewatch.Distance_LE = mean(Applewatch.Distance_LE),
    sd_Applewatch.Distance_LE = sd(Applewatch.Distance_LE),
    
    mean_Fitbit.Steps_LE = mean(Fitbit.Steps_LE),
    sd_Fitbit.Steps_LE = sd(Fitbit.Steps_LE),
    mean_Fitbit.Heart_LE = mean(Fitbit.Heart_LE),
    sd_Fitbit.Heart_LE = sd(Fitbit.Heart_LE),
    mean_Fitbit.Calories_LE = mean(Fitbit.Calories_LE),
    sd_Fitbit.Calories_LE = sd(Fitbit.Calories_LE),
    mean_Fitbit.Distance_LE = mean(Fitbit.Distance_LE),
    sd_Fitbit.Distance_LE = sd(Fitbit.Distance_LE)
  ) %>% 
  mutate(
    count = gender_stats$freq,
    percentage = gender_stats$percent,
  )
```

```
## `summarise()` ungrouping output (override with `.groups` argument)
```

```r
paper_stats <- aggregated_data %>% 
 # group_by(id) %>% 
  summarise(
    mean_Applewatch.Steps_LE = mean(Applewatch.Steps_LE),
    sd_Applewatch.Steps_LE = sd(Applewatch.Steps_LE),
    mean_Applewatch.Heart_LE = mean(Applewatch.Heart_LE),
    sd_Applewatch.Heart_LE = sd(Applewatch.Heart_LE),
    mean_AppleWatch_EE = mean(Applewatch.Calories_LE, na.rm = T),
    sd_AppleWatch_EE = sd(Applewatch.Calories_LE),
    mean_Applewatch.Distance_LE = mean(Applewatch.Distance_LE),
    sd_Applewatch.Distance_LE = sd(Applewatch.Distance_LE),
    mean_EntropyApplewatchHeartPerDay_LE = mean(EntropyApplewatchHeartPerDay_LE),
    sd_EntropyApplewatchHeartPerDay_LE = sd(EntropyApplewatchHeartPerDay_LE),
    mean_EntropyApplewatchStepsPerDay_LE = mean(EntropyApplewatchStepsPerDay_LE),
    sd_EntropyApplewatchStepsPerDay_LE = sd(EntropyApplewatchStepsPerDay_LE),
    mean_RestingApplewatchHeartrate_LE = mean(RestingApplewatchHeartrate_LE),
    sd_RestingApplewatchHeartrate_LE = sd(RestingApplewatchHeartrate_LE),
    mean_CorrelationApplewatchHeartrateSteps_LE = mean(CorrelationApplewatchHeartrateSteps_LE, na.rm = T),
    sd_CorrelationApplewatchHeartrateSteps_LE = sd(CorrelationApplewatchHeartrateSteps_LE, na.rm = T),
    mean_NormalizedApplewatchHeartrate_LE = mean(NormalizedApplewatchHeartrate_LE),
    sd_NormalizedApplewatchHeartrate_LE = sd(NormalizedApplewatchHeartrate_LE),
    mean_ApplewatchIntensity_LE = mean(ApplewatchIntensity_LE),
    sd_ApplewatchIntensity_LE = sd(ApplewatchIntensity_LE),
    mean_SDNormalizedApplewatchHR_LE = mean(SDNormalizedApplewatchHR_LE),
    sd_SDNormalizedApplewatchHR_LE = sd(SDNormalizedApplewatchHR_LE),
    mean_ApplewatchStepsXDistance_LE = mean(ApplewatchStepsXDistance_LE),
    sd_ApplewatchStepsXDistance_LE = sd(ApplewatchStepsXDistance_LE),
    
    mean_Fitbit.Steps_LE = mean(Fitbit.Steps_LE),
    sd_Fitbit.Steps_LE = sd(Fitbit.Steps_LE),
    mean_Fitbit.Heart_LE = mean(Fitbit.Heart_LE),
    sd_Fitbit.Heart_LE = sd(Fitbit.Heart_LE),
    mean_Fitbit_EE = mean(Fitbit.Calories_LE, na.rm = T),
    sd_Fitbit_EE = sd(Fitbit.Calories_LE),
    mean_Fitbit.Distance_LE = mean(Fitbit.Distance_LE),
    sd_Fitbit.Distance_LE = sd(Fitbit.Distance_LE),
    mean_EntropyFitbitHeartPerDay_LE = mean(EntropyFitbitHeartPerDay_LE),
    sd_EntropyFitbitHeartPerDay_LE = sd(EntropyFitbitHeartPerDay_LE),
    mean_EntropyFitbitStepsPerDay_LE = mean(EntropyFitbitStepsPerDay_LE),
    sd_EntropyFitbitStepsPerDay_LE = sd(EntropyFitbitStepsPerDay_LE),
    mean_RestingFitbitHeartrate_LE = mean(RestingFitbitHeartrate_LE),
    sd_RestingFitbitHeartrate_LE = sd(RestingFitbitHeartrate_LE),
    mean_CorrelationFitbitHeartrateSteps_LE = mean(CorrelationFitbitHeartrateSteps_LE, na.rm = T),
    sd_CorrelationFitbitHeartrateSteps_LE = sd(CorrelationFitbitHeartrateSteps_LE, na.rm = T),
    mean_NormalizedFitbitHeartrate_LE = mean(NormalizedFitbitHeartrate_LE),
    sd_NormalizedFitbitHeartrate_LE = sd(NormalizedFitbitHeartrate_LE),
    mean_FitbitIntensity_LE = mean(FitbitIntensity_LE),
    sd_FitbitIntensity_LE = sd(FitbitIntensity_LE),
    mean_SDNormalizedFitbitHR_LE = mean(SDNormalizedFitbitHR_LE),
    sd_SDNormalizedFitbitHR_LE = sd(SDNormalizedFitbitHR_LE),
    mean_FitbitStepsXDistance_LE = mean(FitbitStepsXDistance_LE),
    sd_FitbitStepsXDistance_LE = sd(FitbitStepsXDistance_LE)
  )

paper_stats <- round(paper_stats, 1)

write.csv(paper_stats, "descriptive_stats.csv")  
```

# Apple Watch Data

## Interpolated Results (Included in the paper)

### Decision Tree (Apple Watch)


```r
x_columns_AW_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Applewatch.Steps_LE",
  "Applewatch.Heart_LE",
  "Applewatch.Calories_LE",
  "Applewatch.Distance_LE",
  "EntropyApplewatchHeartPerDay_LE",
  "EntropyApplewatchStepsPerDay_LE",
  "RestingApplewatchHeartrate_LE",
  "CorrelationApplewatchHeartrateSteps_LE",
  "NormalizedApplewatchHeartrate_LE",
  "ApplewatchIntensity_LE",
  "SDNormalizedApplewatchHR_LE",
  "ApplewatchStepsXDistance_LE",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_AW_LE)
```

```
## Note: Using an external vector in selections is ambiguous.
## ℹ Use `all_of(x_columns_AW_LE)` instead of `x_columns_AW_LE` to silence this message.
## ℹ See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
## This message is displayed once per session.
```

```r
write.csv(x, "aw_data_le_21_08_2019.csv", na = "")
x$gender <- ifelse(x$gender == "Male", 1, 0)

idx <- which(is.na(x), arr.ind = T)[, 1]
x <- x[-idx, ]

y <- as.factor(aggregated_data$activity_trimmed[-idx])

# y <- aggregated_data$activity_trimmed

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

train = x[train,]
train$activity_trimmed <- as.factor(train$activity_trimmed)

test = x[test,]
test$activity_trimmed <- as.factor(test$activity_trimmed)


dtModel <- rpart(activity_trimmed ~ ., train)

predict_unseen_aw_le <- predict(dtModel, test, type = 'class')

confusionMatrix(predict_unseen_aw_le, test$activity_trimmed)
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            122             39             12             41
##   Running 3 METs    20             61             12              1
##   Running 5 METs    25             15             97             32
##   Running 7 METs     5              2             11             84
##   Self Pace walk    25             37             13              1
##   Sitting           31             23             28             19
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      34      63
##   Running 3 METs             38       9
##   Running 5 METs              5      24
##   Running 7 METs              4       8
##   Self Pace walk             70      13
##   Sitting                    20      53
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4439          
##                  95% CI : (0.4143, 0.4739)
##     No Information Rate : 0.2078          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3283          
##                                           
##  Mcnemar's Test P-Value : 7.544e-11       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.5351               0.34463               0.56069
## Specificity                0.7825               0.91304               0.89069
## Pos Pred Value             0.3923               0.43262               0.48990
## Neg Pred Value             0.8651               0.87866               0.91546
## Prevalence                 0.2078               0.16135               0.15770
## Detection Rate             0.1112               0.05561               0.08842
## Detection Prevalence       0.2835               0.12853               0.18049
## Balanced Accuracy          0.6588               0.62884               0.72569
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.47191               0.40936        0.31176
## Specificity                        0.96736               0.90389        0.86947
## Pos Pred Value                     0.73684               0.44025        0.30460
## Neg Pred Value                     0.90437               0.89232        0.87324
## Prevalence                         0.16226               0.15588        0.15497
## Detection Rate                     0.07657               0.06381        0.04831
## Detection Prevalence               0.10392               0.14494        0.15861
## Balanced Accuracy                  0.71963               0.65662        0.59062
```

### Random Forest


```r
x_columns_AW_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Applewatch.Steps_LE",
  "Applewatch.Heart_LE",
  "Applewatch.Calories_LE",
  "Applewatch.Distance_LE",
  "EntropyApplewatchHeartPerDay_LE",
  "EntropyApplewatchStepsPerDay_LE",
  "RestingApplewatchHeartrate_LE",
  "CorrelationApplewatchHeartrateSteps_LE",
  "NormalizedApplewatchHeartrate_LE",
  "ApplewatchIntensity_LE",
  "SDNormalizedApplewatchHR_LE",
  "ApplewatchStepsXDistance_LE"
 # ,"activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_AW_LE)
x$gender <- ifelse(x$gender == "Male", 1, 0)

idx <- which(is.na(x), arr.ind = T)[, 1]
x <- x[-idx, ]

y <- as.factor(aggregated_data$activity_trimmed[-idx])

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

xTrain = x[train, ]
yTrain = y[train]
xTest = x[test, ]
yTest = y[test]

rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 100)

### Training 
print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            427             38             27              9
##   Running 3 METs    41            340              9              1
##   Running 5 METs    13              6            343             17
##   Running 7 METs     9              0             20            392
##   Self Pace walk    21             11              4              0
##   Sitting           41              5             17             13
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      17      38
##   Running 3 METs             12      12
##   Running 5 METs              8      30
##   Running 7 METs              0      18
##   Self Pace walk            321      24
##   Sitting                    22     253
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8113          
##                  95% CI : (0.7955, 0.8262)
##     No Information Rate : 0.2157          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7726          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7736                0.8500                0.8167
## Specificity                0.9357                0.9653                0.9654
## Pos Pred Value             0.7680                0.8193                0.8225
## Neg Pred Value             0.9376                0.9720                0.9641
## Prevalence                 0.2157                0.1563                0.1641
## Detection Rate             0.1669                0.1329                0.1340
## Detection Prevalence       0.2173                0.1622                0.1630
## Balanced Accuracy          0.8546                0.9076                0.8910
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9074                0.8447        0.67467
## Specificity                         0.9779                0.9725        0.95513
## Pos Pred Value                      0.8929                0.8425        0.72080
## Neg Pred Value                      0.9811                0.9729        0.94475
## Prevalence                          0.1688                0.1485        0.14654
## Detection Rate                      0.1532                0.1254        0.09887
## Detection Prevalence                0.1716                0.1489        0.13716
## Balanced Accuracy                   0.9427                0.9086        0.81490
```

```r
### Testing 
pred <- predict(rfModel, xTest)
print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            185             20             10              6
##   Running 3 METs    14            141              2              1
##   Running 5 METs     6              6            160              3
##   Running 7 METs     2              1              5            166
##   Self Pace walk    10              3              2              0
##   Sitting           18              1              4              5
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       7      17
##   Running 3 METs              4       0
##   Running 5 METs              4       9
##   Running 7 METs              0      10
##   Self Pace walk            122       4
##   Sitting                    12     137
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8304          
##                  95% CI : (0.8069, 0.8522)
##     No Information Rate : 0.2142          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7955          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7872                0.8198                0.8743
## Specificity                0.9304                0.9773                0.9694
## Pos Pred Value             0.7551                0.8704                0.8511
## Neg Pred Value             0.9413                0.9668                0.9747
## Prevalence                 0.2142                0.1568                0.1668
## Detection Rate             0.1686                0.1285                0.1459
## Detection Prevalence       0.2233                0.1477                0.1714
## Balanced Accuracy          0.8588                0.8985                0.9218
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9171                0.8188         0.7740
## Specificity                         0.9803                0.9800         0.9565
## Pos Pred Value                      0.9022                0.8652         0.7740
## Neg Pred Value                      0.9836                0.9718         0.9565
## Prevalence                          0.1650                0.1358         0.1613
## Detection Rate                      0.1513                0.1112         0.1249
## Detection Prevalence                0.1677                0.1285         0.1613
## Balanced Accuracy                   0.9487                0.8994         0.8653
```
 
### SVM (Apple Watch)

```r
svmModel <- svm(xTrain, yTrain)

### Training
print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            374             91             43             35
##   Running 3 METs    66            208             31             13
##   Running 5 METs    32             34            278             25
##   Running 7 METs    13              9             26            338
##   Self Pace walk    46             47             23              0
##   Sitting           21             11             19             21
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      93      91
##   Running 3 METs             36      16
##   Running 5 METs             26      54
##   Running 7 METs             13      21
##   Self Pace walk            205      41
##   Sitting                     7     152
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6077          
##                  95% CI : (0.5884, 0.6266)
##     No Information Rate : 0.2157          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5247          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.6775               0.52000                0.6619
## Specificity                0.8241               0.92497                0.9201
## Pos Pred Value             0.5144               0.56216                0.6192
## Neg Pred Value             0.9028               0.91229                0.9327
## Prevalence                 0.2157               0.15631                0.1641
## Detection Rate             0.1462               0.08128                0.1086
## Detection Prevalence       0.2841               0.14459                0.1755
## Balanced Accuracy          0.7508               0.72248                0.7910
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.7824               0.53947        0.40533
## Specificity                         0.9614               0.92795        0.96383
## Pos Pred Value                      0.8048               0.56630        0.65801
## Neg Pred Value                      0.9561               0.92035        0.90421
## Prevalence                          0.1688               0.14850        0.14654
## Detection Rate                      0.1321               0.08011        0.05940
## Detection Prevalence                0.1641               0.14146        0.09027
## Balanced Accuracy                   0.8719               0.73371        0.68458
```

```r
### Testing
pred <- predict(svmModel, xTest)
print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            126             44             33             23
##   Running 3 METs    44             73             13              3
##   Running 5 METs    14             24            105             11
##   Running 7 METs     9              3             11            127
##   Self Pace walk    33             22              8              2
##   Sitting            9              6             13             15
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      32      51
##   Running 3 METs             24       8
##   Running 5 METs             12      24
##   Running 7 METs              5      18
##   Self Pace walk             68      22
##   Sitting                     8      54
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5041          
##                  95% CI : (0.4741, 0.5341)
##     No Information Rate : 0.2142          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4             
##                                           
##  Mcnemar's Test P-Value : 3.705e-07       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.5362               0.42442               0.57377
## Specificity                0.7877               0.90054               0.90700
## Pos Pred Value             0.4078               0.44242               0.55263
## Neg Pred Value             0.8617               0.89378               0.91400
## Prevalence                 0.2142               0.15679               0.16682
## Detection Rate             0.1149               0.06655               0.09572
## Detection Prevalence       0.2817               0.15041               0.17320
## Balanced Accuracy          0.6619               0.66248               0.74039
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.7017               0.45638        0.30508
## Specificity                         0.9498               0.90823        0.94457
## Pos Pred Value                      0.7341               0.43871        0.51429
## Neg Pred Value                      0.9416               0.91401        0.87601
## Prevalence                          0.1650               0.13582        0.16135
## Detection Rate                      0.1158               0.06199        0.04923
## Detection Prevalence                0.1577               0.14129        0.09572
## Balanced Accuracy                   0.8257               0.68230        0.62482
```

### Rotation Forest (Apple Watch)

Rotation Forest models were run in Weka and are available in the Github Repo. This code will only run if you have Weka [https://www.cs.waikato.ac.nz/ml/weka/](https://www.cs.waikato.ac.nz/ml/weka/) installed on your system.


```r
x_columns_AW_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Applewatch.Steps_LE",
  "Applewatch.Heart_LE",
  "Applewatch.Calories_LE",
  "Applewatch.Distance_LE",
  "EntropyApplewatchHeartPerDay_LE",
  "EntropyApplewatchStepsPerDay_LE",
  "RestingApplewatchHeartrate_LE",
  "CorrelationApplewatchHeartrateSteps_LE",
  "NormalizedApplewatchHeartrate_LE",
  "ApplewatchIntensity_LE",
  "SDNormalizedApplewatchHR_LE",
  "ApplewatchStepsXDistance_LE",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_AW_LE)
x$gender <- ifelse(x$gender == "Male", 1, 0)

idx <- which(is.na(x), arr.ind = T)[, 1]
x <- x[-idx, ]


y <- aggregated_data$activity_trimmed

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

train = x[train,]
train$activity_trimmed <- as.factor(train$activity_trimmed)

test = x[test,]
test$activity_trimmed <- as.factor(test$activity_trimmed)


#load the RotationForest package as it's not from the default weka packages (note that the package should be installed before on weka using weka package manager)
WPM("load-package", "RotationForest")

# load the RotationForest classifier using RWeka
rotation_forest <- make_Weka_classifier("weka/classifiers/meta/RotationForest")

# generate the RotationForest model
rotationForestModel <- rotation_forest(activity_trimmed ~ ., train)

# predict the results
rotationForestPredict <- predict(rotationForestModel, test, type = 'class')

print(confusionMatrix(rotationForestPredict, test$activity_trimmed))
```


# FitBit Data

## Interpolated Results (Included in the paper)

### Decision Tree (Fitbit)

```r
x_columns_FB_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Fitbit.Steps_LE",
  "Fitbit.Heart_LE",
  "Fitbit.Calories_LE",
  "Fitbit.Distance_LE",
  "EntropyFitbitHeartPerDay_LE",
  "EntropyFitbitStepsPerDay_LE",
  "RestingFitbitHeartrate_LE",
  "CorrelationFitbitHeartrateSteps_LE",
  "NormalizedFitbitHeartrate_LE",
  "FitbitIntensity_LE",
  "SDNormalizedFitbitHR_LE",
  "FitbitStepsXDistance_LE",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_FB_LE)
```

```
## Note: Using an external vector in selections is ambiguous.
## ℹ Use `all_of(x_columns_FB_LE)` instead of `x_columns_FB_LE` to silence this message.
## ℹ See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
## This message is displayed once per session.
```

```r
write_csv(x, "fb_data_le_21_08_2019.csv", na = "")
x$gender <- ifelse(x$gender == "Male", 1, 0)

y <- aggregated_data$activity_trimmed

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

train = x[train,]
train$activity_trimmed <- as.factor(train$activity_trimmed)

test = x[test,]
test$activity_trimmed <- as.factor(test$activity_trimmed)


dtModel <- rpart(activity_trimmed ~ ., train)
predict_unseen_fb_le <- predict(dtModel, test, type = 'class')

print(confusionMatrix(predict_unseen_fb_le, test$activity_trimmed))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            197             10              1              1
##   Running 3 METs    19            114             20              2
##   Running 5 METs     4             35            121             14
##   Running 7 METs     4              5             10            154
##   Self Pace walk     7              4              8              3
##   Sitting           25              5             10             13
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       6      52
##   Running 3 METs             30       5
##   Running 5 METs             20       5
##   Running 7 METs              4       2
##   Self Pace walk             74      10
##   Sitting                    11      97
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6869          
##                  95% CI : (0.6586, 0.7142)
##     No Information Rate : 0.2323          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6209          
##                                           
##  Mcnemar's Test P-Value : 8.435e-07       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7695                0.6590                0.7118
## Specificity                0.9173                0.9182                0.9163
## Pos Pred Value             0.7378                0.6000                0.6080
## Neg Pred Value             0.9293                0.9353                0.9457
## Prevalence                 0.2323                0.1570                0.1543
## Detection Rate             0.1788                0.1034                0.1098
## Detection Prevalence       0.2423                0.1724                0.1806
## Balanced Accuracy          0.8434                0.7886                0.8140
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8235               0.51034        0.56725
## Specificity                         0.9727               0.96656        0.93126
## Pos Pred Value                      0.8603               0.69811        0.60248
## Neg Pred Value                      0.9642               0.92871        0.92136
## Prevalence                          0.1697               0.13158        0.15517
## Detection Rate                      0.1397               0.06715        0.08802
## Detection Prevalence                0.1624               0.09619        0.14610
## Balanced Accuracy                   0.8981               0.73845        0.74925
```

### Random Forest (Fitbit)


```r
x_columns_FB_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Fitbit.Steps_LE",
  "Fitbit.Heart_LE",
  "Fitbit.Calories_LE",
  "Fitbit.Distance_LE",
  "EntropyFitbitHeartPerDay_LE",
  "EntropyFitbitStepsPerDay_LE",
  "RestingFitbitHeartrate_LE",
  "CorrelationFitbitHeartrateSteps_LE",
  "NormalizedFitbitHeartrate_LE",
  "FitbitIntensity_LE",
  "SDNormalizedFitbitHR_LE",
  "FitbitStepsXDistance_LE"
#  ,"activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_FB_LE)
x$gender <- ifelse(x$gender == "Male", 1, 0)


idx <- which(is.na(x), arr.ind = T)[, 1]
x <- x[-idx, ]

# write.csv(x, "data_for_weka_fb.csv")


y <- as.factor(aggregated_data$activity_trimmed[-idx])

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

xTrain = x[train, ]
yTrain = y[train]
xTest = x[test, ]
yTest = y[test]

rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 100)

### Training
print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            353             17              9              4
##   Running 3 METs    20            229              6              0
##   Running 5 METs    11              4            252              2
##   Running 7 METs     7              2              0            335
##   Self Pace walk    11              1              0              2
##   Sitting           12              0             12              6
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      10      15
##   Running 3 METs              5       1
##   Running 5 METs              4      13
##   Running 7 METs              1       6
##   Self Pace walk            233       4
##   Sitting                     4     234
## 
## Overall Statistics
##                                         
##                Accuracy : 0.8964        
##                  95% CI : (0.8815, 0.91)
##     No Information Rate : 0.2268        
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.8748        
##                                         
##  Mcnemar's Test P-Value : 0.5193        
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8527                0.9051                0.9032
## Specificity                0.9610                0.9796                0.9780
## Pos Pred Value             0.8652                0.8774                0.8811
## Neg Pred Value             0.9570                0.9847                0.9825
## Prevalence                 0.2268                0.1386                0.1529
## Detection Rate             0.1934                0.1255                0.1381
## Detection Prevalence       0.2236                0.1430                0.1567
## Balanced Accuracy          0.9068                0.9424                0.9406
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9599                0.9066         0.8571
## Specificity                         0.9892                0.9885         0.9781
## Pos Pred Value                      0.9544                0.9283         0.8731
## Neg Pred Value                      0.9905                0.9848         0.9750
## Prevalence                          0.1912                0.1408         0.1496
## Detection Rate                      0.1836                0.1277         0.1282
## Detection Prevalence                0.1923                0.1375         0.1468
## Balanced Accuracy                   0.9745                0.9476         0.9176
```

```r
### Testing
pred <- predict(rfModel, xTest)
print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            161              3              3              0
##   Running 3 METs     6            111              0              0
##   Running 5 METs     1              4            109              3
##   Running 7 METs     3              0              2            144
##   Self Pace walk     5              6              2              1
##   Sitting            2              1              4              4
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       4       5
##   Running 3 METs              0       0
##   Running 5 METs              0       5
##   Running 7 METs              1       5
##   Self Pace walk             96       4
##   Sitting                     2      86
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9029        
##                  95% CI : (0.88, 0.9228)
##     No Information Rate : 0.2273        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.8826        
##                                         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.9045                0.8880                0.9083
## Specificity                0.9752                0.9909                0.9804
## Pos Pred Value             0.9148                0.9487                0.8934
## Neg Pred Value             0.9720                0.9790                0.9834
## Prevalence                 0.2273                0.1596                0.1533
## Detection Rate             0.2056                0.1418                0.1392
## Detection Prevalence       0.2248                0.1494                0.1558
## Balanced Accuracy          0.9399                0.9394                0.9444
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9474                0.9320         0.8190
## Specificity                         0.9826                0.9735         0.9808
## Pos Pred Value                      0.9290                0.8421         0.8687
## Neg Pred Value                      0.9873                0.9895         0.9722
## Prevalence                          0.1941                0.1315         0.1341
## Detection Rate                      0.1839                0.1226         0.1098
## Detection Prevalence                0.1980                0.1456         0.1264
## Balanced Accuracy                   0.9650                0.9528         0.8999
```

  
### SVM (Fitbit)

```r
svmModel <- svm(xTrain, yTrain)
  
### Training
print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            340             36             27             57
##   Running 3 METs    26             91             22              4
##   Running 5 METs     6             78            218             55
##   Running 7 METs    26              2              8            226
##   Self Pace walk    15             45              3              3
##   Sitting            1              1              1              4
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      35     162
##   Running 3 METs             50      29
##   Running 5 METs             42      23
##   Running 7 METs              3      24
##   Self Pace walk            126      11
##   Sitting                     1      24
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5616          
##                  95% CI : (0.5385, 0.5846)
##     No Information Rate : 0.2268          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4638          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8213               0.35968                0.7814
## Specificity                0.7753               0.91667                0.8680
## Pos Pred Value             0.5175               0.40991                0.5166
## Neg Pred Value             0.9366               0.89894                0.9565
## Prevalence                 0.2268               0.13863                0.1529
## Detection Rate             0.1863               0.04986                0.1195
## Detection Prevalence       0.3600               0.12164                0.2312
## Balanced Accuracy          0.7983               0.63818                0.8247
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.6476               0.49027        0.08791
## Specificity                         0.9573               0.95089        0.99485
## Pos Pred Value                      0.7820               0.62069        0.75000
## Neg Pred Value                      0.9199               0.91924        0.86113
## Prevalence                          0.1912               0.14082        0.14959
## Detection Rate                      0.1238               0.06904        0.01315
## Detection Prevalence                0.1584               0.11123        0.01753
## Balanced Accuracy                   0.8024               0.72058        0.54138
```

```r
### Testing
pred <- predict(svmModel, xTest)
print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            141             14             20             32
##   Running 3 METs    10             35             13              6
##   Running 5 METs     6             48             77             21
##   Running 7 METs    11              3              5             90
##   Self Pace walk     9             24              3              2
##   Sitting            1              1              2              1
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      16      67
##   Running 3 METs             28       4
##   Running 5 METs             22       9
##   Running 7 METs              1      15
##   Self Pace walk             35       6
##   Sitting                     1       4
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4879          
##                  95% CI : (0.4523, 0.5235)
##     No Information Rate : 0.2273          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3712          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7921                0.2800               0.64167
## Specificity                0.7537                0.9073               0.84012
## Pos Pred Value             0.4862                0.3646               0.42077
## Neg Pred Value             0.9249                0.8690               0.92833
## Prevalence                 0.2273                0.1596               0.15326
## Detection Rate             0.1801                0.0447               0.09834
## Detection Prevalence       0.3704                0.1226               0.23372
## Balanced Accuracy          0.7729                0.5936               0.74089
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.5921                0.3398       0.038095
## Specificity                         0.9445                0.9353       0.991150
## Pos Pred Value                      0.7200                0.4430       0.400000
## Neg Pred Value                      0.9058                0.9034       0.869340
## Prevalence                          0.1941                0.1315       0.134100
## Detection Rate                      0.1149                0.0447       0.005109
## Detection Prevalence                0.1596                0.1009       0.012771
## Balanced Accuracy                   0.7683                0.6375       0.514623
```

### Rotation Forest (Fitbit)

Rotation Forest models were run in Weka and are available in the Github Repo. This code will only run if you have Weka [https://www.cs.waikato.ac.nz/ml/weka/](https://www.cs.waikato.ac.nz/ml/weka/) installed on your system.


```r
x_columns_FB_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Fitbit.Steps_LE",
  "Fitbit.Heart_LE",
  "Fitbit.Calories_LE",
  "Fitbit.Distance_LE",
  "EntropyFitbitHeartPerDay_LE",
  "EntropyFitbitStepsPerDay_LE",
  "RestingFitbitHeartrate_LE",
  "CorrelationFitbitHeartrateSteps_LE",
  "NormalizedFitbitHeartrate_LE",
  "FitbitIntensity_LE",
  "SDNormalizedFitbitHR_LE",
  "FitbitStepsXDistance_LE",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_FB_LE)
x$gender <- ifelse(x$gender == "Male", 1, 0)

idx <- which(is.na(x), arr.ind = T)[, 1]
x <- x[-idx, ]


y <- aggregated_data$activity_trimmed

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

train = x[train,]
train$activity_trimmed <- as.factor(train$activity_trimmed)

test = x[test,]
test$activity_trimmed <- as.factor(test$activity_trimmed)


#load the RotationForest package
WPM("load-package", "RotationForest")

# load the RotationForest classifier using RWeka
rotation_forest <- make_Weka_classifier("weka/classifiers/meta/RotationForest")

# generate the RotationForest model
rotationForestModel <- rotation_forest(activity_trimmed ~ ., train)

# predict the results
rotationForestPredict <- predict(rotationForestModel, test, type = 'class')

print(confusionMatrix(rotationForestPredict, test$activity_trimmed))
```


# Including device type as a feature

## Device name (Fitbit or Apple Watch) as a feature  

### Random Forest

```r
aggregated_data_appended <- read_csv("aggregated_fitbit_applewatch_jaeger_appended.csv")
```

```
## Warning: Missing column names filled in: 'X1' [1]
```

```
## Parsed with column specification:
## cols(
##   .default = col_double(),
##   Username = col_character(),
##   DeviceName = col_character(),
##   DateTime = col_datetime(format = ""),
##   gender = col_character(),
##   activity = col_character(),
##   activity_trimmed = col_character()
## )
```

```
## See spec(...) for full column specifications.
```

```r
x_columns_FB_LE <- c(
  "DeviceName",
  "age",
  "gender",
  "height",
  "weight",
  "Steps_LE",
  "Heart_LE",
  "Calories_LE",
  "Distance_LE",
  "EntropyHeartPerDay_LE",
  "EntropyStepsPerDay_LE",
  "RestingHeartrate_LE",
  "CorrelationHeartrateSteps_LE",
  "NormalizedHeartrate_LE",
  "Intensity_LE",
  "SDNormalizedHR_LE",
  "StepsXDistance_LE"
  #,"activity_trimmed"
)

x <- aggregated_data_appended %>% select(x_columns_FB_LE)
x$gender <- ifelse(x$gender == "Male", 1, 0)
x$DeviceName <- ifelse(x$DeviceName == "AppleWatch", 1, 0)


idx <- which(is.na(x), arr.ind = T)[, 1]
x <- x[-idx, ]

write.csv(x, "data_for_process.csv")


y <- as.factor(aggregated_data_appended$activity_trimmed[-idx])

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

xTrain = x[train, ]
yTrain = y[train]
xTest = x[test, ]
yTest = y[test]

rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 100)

### Training
print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            794             54             41             20
##   Running 3 METs    51            585             12              3
##   Running 5 METs    28             10            617             23
##   Running 7 METs    15              2             16            683
##   Self Pace walk    37              8              4              1
##   Sitting           50              7             23             30
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      23      44
##   Running 3 METs             17      11
##   Running 5 METs             12      40
##   Running 7 METs              1      27
##   Self Pace walk            539      21
##   Sitting                    29     506
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8495          
##                  95% CI : (0.8385, 0.8599)
##     No Information Rate : 0.2224          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.8184          
##                                           
##  Mcnemar's Test P-Value : 0.09092         
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8144                0.8784                0.8654
## Specificity                0.9466                0.9747                0.9692
## Pos Pred Value             0.8135                0.8616                0.8452
## Neg Pred Value             0.9469                0.9781                0.9737
## Prevalence                 0.2224                0.1519                0.1626
## Detection Rate             0.1811                0.1334                0.1407
## Detection Prevalence       0.2226                0.1549                0.1665
## Balanced Accuracy          0.8805                0.9265                0.9173
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8987                0.8680         0.7797
## Specificity                         0.9832                0.9811         0.9628
## Pos Pred Value                      0.9180                0.8836         0.7845
## Neg Pred Value                      0.9788                0.9783         0.9618
## Prevalence                          0.1734                0.1417         0.1480
## Detection Rate                      0.1558                0.1229         0.1154
## Detection Prevalence                0.1697                0.1391         0.1471
## Balanced Accuracy                   0.9409                0.9245         0.8712
```

```r
### Testing
pred <- predict(rfModel, xTest)
print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            329             20              7              6
##   Running 3 METs    26            252              8              0
##   Running 5 METs     8              3            253              6
##   Running 7 METs     3              1             12            333
##   Self Pace walk    13              8              2              0
##   Sitting           25              0              7              9
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      14      25
##   Running 3 METs              6       2
##   Running 5 METs              0      20
##   Running 7 METs              1      14
##   Self Pace walk            237      12
##   Sitting                    10     208
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8574          
##                  95% CI : (0.8408, 0.8729)
##     No Information Rate : 0.2149          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.8281          
##                                           
##  Mcnemar's Test P-Value : 0.173           
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8144                0.8873                0.8754
## Specificity                0.9512                0.9737                0.9767
## Pos Pred Value             0.8204                0.8571                0.8724
## Neg Pred Value             0.9493                0.9798                0.9774
## Prevalence                 0.2149                0.1511                0.1537
## Detection Rate             0.1750                0.1340                0.1346
## Detection Prevalence       0.2133                0.1564                0.1543
## Balanced Accuracy          0.8828                0.9305                0.9261
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9407                0.8843         0.7402
## Specificity                         0.9797                0.9783         0.9681
## Pos Pred Value                      0.9148                0.8713         0.8031
## Neg Pred Value                      0.9861                0.9807         0.9550
## Prevalence                          0.1883                0.1426         0.1495
## Detection Rate                      0.1771                0.1261         0.1106
## Detection Prevalence                0.1936                0.1447         0.1378
## Balanced Accuracy                   0.9602                0.9313         0.8542
```

### SVM


```r
svmModel <- svm(xTrain, yTrain)

### Training
print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            719            161            129            197
##   Running 3 METs    76            241             59             12
##   Running 5 METs    31            133            410             90
##   Running 7 METs    24              5             34            409
##   Self Pace walk    91            105             65             19
##   Sitting           34             21             16             33
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                     156     291
##   Running 3 METs             97      39
##   Running 5 METs             82      57
##   Running 7 METs             13      41
##   Self Pace walk            259      72
##   Sitting                    14     149
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4989          
##                  95% CI : (0.4839, 0.5138)
##     No Information Rate : 0.2224          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3878          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7374               0.36186               0.57504
## Specificity                0.7260               0.92388               0.89294
## Pos Pred Value             0.4350               0.45992               0.51059
## Neg Pred Value             0.9063               0.88990               0.91539
## Prevalence                 0.2224               0.15192               0.16264
## Detection Rate             0.1640               0.05497               0.09352
## Detection Prevalence       0.3771               0.11953               0.18317
## Balanced Accuracy          0.7317               0.64287               0.73399
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.53816               0.41707        0.22958
## Specificity                        0.96772               0.90646        0.96841
## Pos Pred Value                     0.77757               0.42390        0.55805
## Neg Pred Value                     0.90902               0.90406        0.87855
## Prevalence                         0.17336               0.14165        0.14804
## Detection Rate                     0.09329               0.05908        0.03399
## Detection Prevalence               0.11998               0.13937        0.06090
## Balanced Accuracy                  0.75294               0.66176        0.59900
```

```r
### Testing
pred <- predict(svmModel, xTest)
print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            283             79             57             96
##   Running 3 METs    43             85             30              7
##   Running 5 METs    15             56            146             45
##   Running 7 METs    14              8             16            180
##   Self Pace walk    34             47             32              8
##   Sitting           15              9              8             18
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      65     136
##   Running 3 METs             54      15
##   Running 5 METs             40      31
##   Running 7 METs             10      33
##   Self Pace walk             96      31
##   Sitting                     3      35
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4388          
##                  95% CI : (0.4162, 0.4616)
##     No Information Rate : 0.2149          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3158          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7005               0.29930               0.50519
## Specificity                0.7066               0.90664               0.88246
## Pos Pred Value             0.3953               0.36325               0.43844
## Neg Pred Value             0.8960               0.87910               0.90756
## Prevalence                 0.2149               0.15106               0.15372
## Detection Rate             0.1505               0.04521               0.07766
## Detection Prevalence       0.3809               0.12447               0.17713
## Balanced Accuracy          0.7036               0.60297               0.69383
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.50847               0.35821        0.12456
## Specificity                        0.94692               0.90571        0.96685
## Pos Pred Value                     0.68966               0.38710        0.39773
## Neg Pred Value                     0.89253               0.89461        0.86272
## Prevalence                         0.18830               0.14255        0.14947
## Detection Rate                     0.09574               0.05106        0.01862
## Detection Prevalence               0.13883               0.13191        0.04681
## Balanced Accuracy                  0.72770               0.63196        0.54570
```


## Generating AppleWatch and Fitbit data for Rotation Forest model in Weka

```r
x_columns_AW_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Applewatch.Steps_LE",
  "Applewatch.Heart_LE",
  "Applewatch.Calories_LE",
  "Applewatch.Distance_LE",
  "EntropyApplewatchHeartPerDay_LE",
  "EntropyApplewatchStepsPerDay_LE",
  "RestingApplewatchHeartrate_LE",
  "CorrelationApplewatchHeartrateSteps_LE",
  "NormalizedApplewatchHeartrate_LE",
  "ApplewatchIntensity_LE",
  "SDNormalizedApplewatchHR_LE",
  "ApplewatchStepsXDistance_LE",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_AW_LE)
x$gender <- ifelse(x$gender == "Male", 1, 0)

idx <- which(is.na(x), arr.ind = T)[, 1]
x <- x[-idx, ]

write.csv(x, "data_for_weka_aw.csv")

x_columns_FB_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Fitbit.Steps_LE",
  "Fitbit.Heart_LE",
  "Fitbit.Calories_LE",
  "Fitbit.Distance_LE",
  "EntropyFitbitHeartPerDay_LE",
  "EntropyFitbitStepsPerDay_LE",
  "RestingFitbitHeartrate_LE",
  "CorrelationFitbitHeartrateSteps_LE",
  "NormalizedFitbitHeartrate_LE",
  "FitbitIntensity_LE",
  "SDNormalizedFitbitHR_LE",
  "FitbitStepsXDistance_LE",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_FB_LE)
x$gender <- ifelse(x$gender == "Male", 1, 0)


idx <- which(is.na(x), arr.ind = T)[, 1]
x <- x[-idx, ]

write.csv(x, "data_for_weka_fb.csv")
```


## Non-Interpolated Results (Not included in the paper)

### Applewatch with non-interpolated data - Decision tree

```r
x_columns_AW <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Applewatch.Steps",
  "Applewatch.Heart",
  "Applewatch.Calories",
  "Applewatch.Distance",
  "EntropyApplewatchHeartPerDay",
  "EntropyApplewatchStepsPerDay",
  "RestingApplewatchHeartrate",
  "CorrelationApplewatchHeartrateSteps",
  "NormalizedApplewatchHeartrate",
  "ApplewatchIntensity",
  "SDNormalizedApplewatchHR",
  "ApplewatchStepsXDistance",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_AW)
```

```
## Note: Using an external vector in selections is ambiguous.
## ℹ Use `all_of(x_columns_AW)` instead of `x_columns_AW` to silence this message.
## ℹ See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
## This message is displayed once per session.
```

```r
x$gender <- ifelse(x$gender == "Male", 1, 0)

y <- aggregated_data$activity_trimmed

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

train = x[train,]
train$activity_trimmed <- as.factor(train$activity_trimmed)

test = x[test,]
test$activity_trimmed <- as.factor(test$activity_trimmed)


dtModel <- rpart(activity_trimmed ~ ., train)
predict_unseen_aw <- predict(dtModel, test, type = 'class')

print(confusionMatrix(predict_unseen_aw, test$activity_trimmed))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            180            105             88             64
##   Running 3 METs     0              0              0              0
##   Running 5 METs    55             33             57             39
##   Running 7 METs    13             20             35             93
##   Self Pace walk     2             14              3              2
##   Sitting            0              0              0              0
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      97     117
##   Running 3 METs              0       0
##   Running 5 METs             26      25
##   Running 7 METs              9       8
##   Self Pace walk             14       3
##   Sitting                     0       0
## 
## Overall Statistics
##                                           
##                Accuracy : 0.3122          
##                  95% CI : (0.2849, 0.3404)
##     No Information Rate : 0.2269          
##     P-Value [Acc > NIR] : 4.536e-11       
##                                           
##                   Kappa : 0.1369          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7200                0.0000               0.31148
## Specificity                0.4472                1.0000               0.80631
## Pos Pred Value             0.2765                   NaN               0.24255
## Neg Pred Value             0.8448                0.8439               0.85467
## Prevalence                 0.2269                0.1561               0.16606
## Detection Rate             0.1633                0.0000               0.05172
## Detection Prevalence       0.5907                0.0000               0.21325
## Balanced Accuracy          0.5836                0.5000               0.55889
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.46970               0.09589         0.0000
## Specificity                        0.90597               0.97490         1.0000
## Pos Pred Value                     0.52247               0.36842            NaN
## Neg Pred Value                     0.88636               0.87594         0.8612
## Prevalence                         0.17967               0.13249         0.1388
## Detection Rate                     0.08439               0.01270         0.0000
## Detection Prevalence               0.16152               0.03448         0.0000
## Balanced Accuracy                  0.68784               0.53539         0.5000
```


##### Fitbit with non-interpolated data - Decision tree

```r
x_columns_FB <- c(
  "age",
  "gender",
  "height",
  "weight",
  "Fitbit.Steps",
  "Fitbit.Heart",
  "Fitbit.Calories",
  "Fitbit.Distance",
  "EntropyFitbitHeartPerDay",
  "EntropyFitbitStepsPerDay",
  "RestingFitbitHeartrate",
  "CorrelationFitbitHeartrateSteps",
  "NormalizedFitbitHeartrate",
  "FitbitIntensity",
  "SDNormalizedFitbitHR",
  "FitbitStepsXDistance",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_FB)
```

```
## Note: Using an external vector in selections is ambiguous.
## ℹ Use `all_of(x_columns_FB)` instead of `x_columns_FB` to silence this message.
## ℹ See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
## This message is displayed once per session.
```

```r
x$gender <- ifelse(x$gender == "Male", 1, 0)

y <- aggregated_data$activity_trimmed

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

train = x[train,]
train$activity_trimmed <- as.factor(train$activity_trimmed)

test = x[test,]
test$activity_trimmed <- as.factor(test$activity_trimmed)


dtModel <- rpart(activity_trimmed ~ ., train)
predict_unseen_fb <- predict(dtModel, test, type = 'class')

print(confusionMatrix(predict_unseen_fb, test$activity_trimmed))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            185             12             14             11
##   Running 3 METs    14             74             19              8
##   Running 5 METs     6             47            136             40
##   Running 7 METs     0              6             11            110
##   Self Pace walk     0              0              0              0
##   Sitting           36             18             12              5
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      16     116
##   Running 3 METs             85       7
##   Running 5 METs             47      16
##   Running 7 METs              1       1
##   Self Pace walk              0       0
##   Sitting                    17      32
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4873          
##                  95% CI : (0.4574, 0.5173)
##     No Information Rate : 0.2187          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3758          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7676               0.47134                0.7083
## Specificity                0.8037               0.85926                0.8286
## Pos Pred Value             0.5226               0.35749                0.4658
## Neg Pred Value             0.9251               0.90726                0.9309
## Prevalence                 0.2187               0.14247                0.1742
## Detection Rate             0.1679               0.06715                0.1234
## Detection Prevalence       0.3212               0.18784                0.2650
## Balanced Accuracy          0.7857               0.66530                0.7685
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.63218                0.0000        0.18605
## Specificity                        0.97953                1.0000        0.90538
## Pos Pred Value                     0.85271                   NaN        0.26667
## Neg Pred Value                     0.93422                0.8494        0.85743
## Prevalence                         0.15789                0.1506        0.15608
## Detection Rate                     0.09982                0.0000        0.02904
## Detection Prevalence               0.11706                0.0000        0.10889
## Balanced Accuracy                  0.80585                0.5000        0.54571
```


