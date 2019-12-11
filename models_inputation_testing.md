---
title: "Classification"
author: "Arastoo Bozorgi"
date: "07/06/2019"
output:
      html_document:
        keep_md: true
---



## Apply SVM classification to predict the labels in the smart_phone accel study

### Loading the required libraries

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
## ── Attaching packages ─────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──
```

```
## ✔ ggplot2 3.2.1     ✔ readr   1.3.1
## ✔ tibble  2.1.3     ✔ purrr   0.3.2
## ✔ tidyr   0.8.3     ✔ stringr 1.4.0
## ✔ ggplot2 3.2.1     ✔ forcats 0.4.0
```

```
## ── Conflicts ────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ dplyr::between()   masks data.table::between()
## ✖ dplyr::filter()    masks stats::filter()
## ✖ dplyr::first()     masks data.table::first()
## ✖ dplyr::lag()       masks stats::lag()
## ✖ dplyr::last()      masks data.table::last()
## ✖ purrr::transpose() masks data.table::transpose()
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
library("RWeka")
```



### Get the os type

```r
OS <- Sys.info()
if (OS["sysname"] == "Windows") {
  path <-
    "Z:/Research/dfuller/Walkabilly/studies/smarphone_accel/data/"
} else {
  path <-
    "/Volumes/hkr-storage/Research/dfuller/Walkabilly/studies/smarphone_accel/data/"
}
setwd(path)
```

### Reading the generated data

```r
aggregated_data <- fread(paste0(path, "aggregated_fitbit_applewatch_jaeger.csv"), data.table = F)[ ,-1]
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


### Applying the SVM on the data

##### Applewatch with non-interpolated data - Decision tree

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
##   Lying            231            173            136            126
##   Running 3 METs     0              0              0              0
##   Running 5 METs     2              4             22             16
##   Running 7 METs     5              4             11             56
##   Self Pace walk     0              0              0              0
##   Sitting            0              0              0              0
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                     153     146
##   Running 3 METs              0       0
##   Running 5 METs              2       1
##   Running 7 METs              8       6
##   Self Pace walk              0       0
##   Sitting                     0       0
## 
## Overall Statistics
##                                          
##                Accuracy : 0.2804         
##                  95% CI : (0.254, 0.3079)
##     No Information Rate : 0.216          
##     P-Value [Acc > NIR] : 2.778e-07      
##                                          
##                   Kappa : 0.0887         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.9706                0.0000
## Specificity                0.1505                1.0000
## Pos Pred Value             0.2394                   NaN
## Neg Pred Value             0.9489                0.8358
## Prevalence                 0.2160                0.1642
## Detection Rate             0.2096                0.0000
## Detection Prevalence       0.8757                0.0000
## Balanced Accuracy          0.5605                0.5000
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                        0.13018               0.28283
## Specificity                        0.97320               0.96239
## Pos Pred Value                     0.46809               0.62222
## Neg Pred Value                     0.86066               0.85968
## Prevalence                         0.15336               0.17967
## Detection Rate                     0.01996               0.05082
## Detection Prevalence               0.04265               0.08167
## Balanced Accuracy                  0.55169               0.62261
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.0000         0.0000
## Specificity                         1.0000         1.0000
## Pos Pred Value                         NaN            NaN
## Neg Pred Value                      0.8521         0.8612
## Prevalence                          0.1479         0.1388
## Detection Rate                      0.0000         0.0000
## Detection Prevalence                0.0000         0.0000
## Balanced Accuracy                   0.5000         0.5000
```

##### Applewatch with interpolated data - Decision tree

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

print(confusionMatrix(predict_unseen_aw_le, test$activity_trimmed))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            134             40             28             38
##   Running 3 METs     4             31             17              0
##   Running 5 METs    18             16             61             14
##   Running 7 METs     2              1             25             95
##   Self Pace walk    79             85             51             17
##   Sitting            8              8              9              4
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      36      63
##   Running 3 METs              4       9
##   Running 5 METs              2       7
##   Running 7 METs              5      15
##   Self Pace walk             99      53
##   Sitting                     3      16
## 
## Overall Statistics
##                                           
##                Accuracy : 0.3974          
##                  95% CI : (0.3683, 0.4271)
##     No Information Rate : 0.2233          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.2727          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.5469               0.17127
## Specificity                0.7594               0.96288
## Pos Pred Value             0.3953               0.47692
## Neg Pred Value             0.8536               0.85465
## Prevalence                 0.2233               0.16500
## Detection Rate             0.1222               0.02826
## Detection Prevalence       0.3090               0.05925
## Balanced Accuracy          0.6532               0.56708
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                        0.31937                0.5655
## Specificity                        0.93709                0.9483
## Pos Pred Value                     0.51695                0.6643
## Neg Pred Value                     0.86721                0.9235
## Prevalence                         0.17411                0.1531
## Detection Rate                     0.05561                0.0866
## Detection Prevalence               0.10757                0.1304
## Balanced Accuracy                  0.62823                0.7569
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.66443        0.09816
## Specificity                        0.69937        0.96574
## Pos Pred Value                     0.25781        0.33333
## Neg Pred Value                     0.92987        0.85987
## Prevalence                         0.13582        0.14859
## Detection Rate                     0.09025        0.01459
## Detection Prevalence               0.35005        0.04376
## Balanced Accuracy                  0.68190        0.53195
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
##   Lying            188             12             10             13
##   Running 3 METs     5             52             17              4
##   Running 5 METs     2             59            142             34
##   Running 7 METs     3              4             14            125
##   Self Pace walk    10             31              6              8
##   Sitting           26             13             10              4
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      15     113
##   Running 3 METs             29       3
##   Running 5 METs             36      14
##   Running 7 METs              1       2
##   Self Pace walk             49       7
##   Sitting                    16      25
## 
## Overall Statistics
##                                          
##                Accuracy : 0.5272         
##                  95% CI : (0.4973, 0.557)
##     No Information Rate : 0.2123         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.424          
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8034               0.30409
## Specificity                0.8122               0.93770
## Pos Pred Value             0.5356               0.47273
## Neg Pred Value             0.9387               0.88004
## Prevalence                 0.2123               0.15517
## Detection Rate             0.1706               0.04719
## Detection Prevalence       0.3185               0.09982
## Balanced Accuracy          0.8078               0.62090
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.7136                0.6649
## Specificity                         0.8394                0.9737
## Pos Pred Value                      0.4948                0.8389
## Neg Pred Value                      0.9301                0.9339
## Prevalence                          0.1806                0.1706
## Detection Rate                      0.1289                0.1134
## Detection Prevalence                0.2604                0.1352
## Balanced Accuracy                   0.7765                0.8193
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.33562        0.15244
## Specificity                        0.93515        0.92644
## Pos Pred Value                     0.44144        0.26596
## Neg Pred Value                     0.90212        0.86210
## Prevalence                         0.13249        0.14882
## Detection Rate                     0.04446        0.02269
## Detection Prevalence               0.10073        0.08530
## Balanced Accuracy                  0.63538        0.53944
```

##### Fitbit with interpolated data - Decision tree

```r
x_columns_FB_LE <- c(
  "age",
  "gender",
  "height",
  "weight",
#  "Fitbit.Steps_LE",
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
##   Lying            189             24             17             16
##   Running 3 METs    13            104             15              6
##   Running 5 METs     4             37            122             20
##   Running 7 METs     1              4             11            156
##   Self Pace walk    12              8              2              2
##   Sitting            3              0              0              0
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      22     101
##   Running 3 METs             49       5
##   Running 5 METs             23      12
##   Running 7 METs              5       2
##   Self Pace walk             68       9
##   Sitting                     4      36
## 
## Overall Statistics
##                                          
##                Accuracy : 0.6125         
##                  95% CI : (0.583, 0.6414)
##     No Information Rate : 0.2015         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.5304         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8514               0.58757
## Specificity                0.7955               0.90486
## Pos Pred Value             0.5122               0.54167
## Neg Pred Value             0.9550               0.91978
## Prevalence                 0.2015               0.16062
## Detection Rate             0.1715               0.09437
## Detection Prevalence       0.3348               0.17423
## Balanced Accuracy          0.8234               0.74622
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.7305                0.7800
## Specificity                         0.8973                0.9745
## Pos Pred Value                      0.5596                0.8715
## Neg Pred Value                      0.9491                0.9523
## Prevalence                          0.1515                0.1815
## Detection Rate                      0.1107                0.1416
## Detection Prevalence                0.1978                0.1624
## Balanced Accuracy                   0.8139                0.8773
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.39766        0.21818
## Specificity                        0.96455        0.99253
## Pos Pred Value                     0.67327        0.83721
## Neg Pred Value                     0.89710        0.87819
## Prevalence                         0.15517        0.14973
## Detection Rate                     0.06171        0.03267
## Detection Prevalence               0.09165        0.03902
## Balanced Accuracy                  0.68111        0.60536
```

##### Applewatch with non-interpolated data with EE - regression

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
  "EE",
  "activity_trimmed"
)

x <- aggregated_data %>% select(x_columns_AW)
x$gender <- ifelse(x$gender == "Male", 1, 0)

y <- aggregated_data$activity_trimmed

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

train = x[train,]
train$activity_trimmed <- as.factor(train$activity_trimmed)

test = x[test,]
test$activity_trimmed <- as.factor(test$activity_trimmed)


dtModel <- rpart(activity_trimmed ~ ., train, method = 'anova')
printcp(dtModel) # display the results 
```

```
## 
## Regression tree:
## rpart(formula = activity_trimmed ~ ., data = train, method = "anova")
## 
## Variables actually used in tree construction:
## [1] ApplewatchIntensity EE                 
## 
## Root node error: 7724.2/2569 = 3.0067
## 
## n= 2569 
## 
##        CP nsplit rel error xerror     xstd
## 1 0.01297      0   1.00000 1.0005 0.016972
## 2 0.01000      2   0.97406 0.9832 0.018539
```

```r
plotcp(dtModel) # visualize cross-validation results 
```

![](Classification_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

```r
summary(dtModel) # detailed summary of splits
```

```
## Call:
## rpart(formula = activity_trimmed ~ ., data = train, method = "anova")
##   n= 2569 
## 
##           CP nsplit rel error    xerror       xstd
## 1 0.01296964      0 1.0000000 1.0005298 0.01697175
## 2 0.01000000      2 0.9740607 0.9831978 0.01853933
## 
## Variable importance
##           ApplewatchIntensity NormalizedApplewatchHeartrate 
##                            31                            29 
##                            EE              Applewatch.Heart 
##                            25                            12 
##  EntropyApplewatchStepsPerDay  EntropyApplewatchHeartPerDay 
##                             1                             1 
## 
## Node number 1: 2569 observations,    complexity param=0.01296964
##   mean=3.291164, MSE=3.006699 
##   left son=2 (422 obs) right son=3 (2147 obs)
##   Primary splits:
##       ApplewatchIntensity           < 0.05181153  to the left,  improve=0.020969500, (1959 missing)
##       NormalizedApplewatchHeartrate < 6.916667    to the left,  improve=0.020225320, (1959 missing)
##       EE                            < 6.088331    to the left,  improve=0.019715590, (2 missing)
##       Applewatch.Heart              < 86.76795    to the left,  improve=0.007588451, (1959 missing)
##       Applewatch.Calories           < 0.8635      to the right, improve=0.004528655, (1074 missing)
##   Surrogate splits:
##       NormalizedApplewatchHeartrate < 6.318182    to the left,  agree=0.985, adj=0.934, (0 split)
##       Applewatch.Heart              < 76.58333    to the left,  agree=0.862, adj=0.387, (0 split)
##       EE                            < 1.208764    to the left,  agree=0.808, adj=0.146, (1958 split)
##       EntropyApplewatchStepsPerDay  < 0.2124835   to the left,  agree=0.785, adj=0.044, (1 split)
##       EntropyApplewatchHeartPerDay  < 0.408991    to the left,  agree=0.782, adj=0.029, (0 split)
## 
## Node number 2: 422 observations
##   mean=2.85545, MSE=5.057304 
## 
## Node number 3: 2147 observations,    complexity param=0.01296964
##   mean=3.376805, MSE=2.558996 
##   left son=6 (1795 obs) right son=7 (352 obs)
##   Primary splits:
##       EE                       < 6.699906    to the left,  improve=0.019067750, (2 missing)
##       Applewatch.Calories      < 0.2413      to the right, improve=0.005627719, (873 missing)
##       Applewatch.Steps         < 5.063406    to the left,  improve=0.005001750, (1251 missing)
##       ApplewatchStepsXDistance < 0.02496125  to the left,  improve=0.004922634, (1255 missing)
##       Applewatch.Distance      < 0.004065833 to the left,  improve=0.004006968, (1119 missing)
## 
## Node number 6: 1795 observations
##   mean=3.279109, MSE=2.957753 
## 
## Node number 7: 352 observations
##   mean=3.875, MSE=0.2286932
```

##### Applewatch with interpolated data - Random Forest and SVM

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


# write.csv(x, "data_for_weka_aw.csv")

y <- as.factor(aggregated_data$activity_trimmed[-idx])

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

xTrain = x[train, ]
yTrain = y[train]
xTest = x[test, ]
yTest = y[test]


print(
    "--------------------------------------------Random Forest--------------------------------------------"
  )
```

```
## [1] "--------------------------------------------Random Forest--------------------------------------------"
```

```r
  rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 500)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            412             44             24             10
##   Running 3 METs    34            322              5              0
##   Running 5 METs    18             10            384             15
##   Running 7 METs     5              0             14            383
##   Self Pace walk    22             15              4              0
##   Sitting           45              3             11             15
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      19      34
##   Running 3 METs             12      11
##   Running 5 METs              7      27
##   Running 7 METs              0      16
##   Self Pace walk            306      15
##   Sitting                    22     295
## 
## Overall Statistics
##                                          
##                Accuracy : 0.8214         
##                  95% CI : (0.806, 0.8361)
##     No Information Rate : 0.2095         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.7849         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.7687                0.8173
## Specificity                0.9352                0.9714
## Pos Pred Value             0.7587                0.8385
## Neg Pred Value             0.9385                0.9669
## Prevalence                 0.2095                0.1540
## Detection Rate             0.1610                0.1258
## Detection Prevalence       0.2122                0.1501
## Balanced Accuracy          0.8520                0.8943
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.8688                0.9054
## Specificity                         0.9636                0.9836
## Pos Pred Value                      0.8330                0.9163
## Neg Pred Value                      0.9724                0.9813
## Prevalence                          0.1727                0.1653
## Detection Rate                      0.1501                0.1497
## Detection Prevalence                0.1801                0.1633
## Balanced Accuracy                   0.9162                0.9445
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8361         0.7412
## Specificity                         0.9745         0.9556
## Pos Pred Value                      0.8453         0.7545
## Neg Pred Value                      0.9727         0.9525
## Prevalence                          0.1430         0.1555
## Detection Rate                      0.1196         0.1153
## Detection Prevalence                0.1415         0.1528
## Balanced Accuracy                   0.9053         0.8484
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(rfModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            204             12              5              6
##   Running 3 METs    12            154              2              2
##   Running 5 METs     8              1            140              7
##   Running 7 METs     2              1              6            168
##   Self Pace walk     7              5              1              0
##   Sitting           18              5              7              7
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       9       8
##   Running 3 METs              3       2
##   Running 5 METs              3      13
##   Running 7 METs              0       6
##   Self Pace walk            133       8
##   Sitting                    15     117
## 
## Overall Statistics
##                                           
##                Accuracy : 0.835           
##                  95% CI : (0.8117, 0.8565)
##     No Information Rate : 0.2288          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8009          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8127                0.8652
## Specificity                0.9527                0.9771
## Pos Pred Value             0.8361                0.8800
## Neg Pred Value             0.9449                0.9740
## Prevalence                 0.2288                0.1623
## Detection Rate             0.1860                0.1404
## Detection Prevalence       0.2224                0.1595
## Balanced Accuracy          0.8827                0.9212
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.8696                0.8842
## Specificity                         0.9658                0.9835
## Pos Pred Value                      0.8140                0.9180
## Neg Pred Value                      0.9773                0.9759
## Prevalence                          0.1468                0.1732
## Detection Rate                      0.1276                0.1531
## Detection Prevalence                0.1568                0.1668
## Balanced Accuracy                   0.9177                0.9338
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8160         0.7597
## Specificity                         0.9775         0.9449
## Pos Pred Value                      0.8636         0.6923
## Neg Pred Value                      0.9682         0.9601
## Prevalence                          0.1486         0.1404
## Detection Rate                      0.1212         0.1067
## Detection Prevalence                0.1404         0.1541
## Balanced Accuracy                   0.8967         0.8523
```

```r
  print(
    "------------------------------------------------SVM--------------------------------------------------"
  )
```

```
## [1] "------------------------------------------------SVM--------------------------------------------------"
```

```r
  svmModel <- svm(xTrain, yTrain)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            318             64             54             56
##   Running 3 METs   113            245             44              5
##   Running 5 METs    23             32            288             34
##   Running 7 METs    11              8             14            301
##   Self Pace walk    36             30             17              3
##   Sitting           35             15             25             24
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      68      80
##   Running 3 METs             84      28
##   Running 5 METs             26      59
##   Running 7 METs             15      20
##   Self Pace walk            159      28
##   Sitting                    14     183
## 
## Overall Statistics
##                                          
##                Accuracy : 0.5838         
##                  95% CI : (0.5644, 0.603)
##     No Information Rate : 0.2095         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.4974         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.5933               0.62183
## Specificity                0.8408               0.87344
## Pos Pred Value             0.4969               0.47206
## Neg Pred Value             0.8864               0.92696
## Prevalence                 0.2095               0.15397
## Detection Rate             0.1243               0.09574
## Detection Prevalence       0.2501               0.20281
## Balanced Accuracy          0.7171               0.74763
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.6516                0.7116
## Specificity                         0.9178                0.9682
## Pos Pred Value                      0.6234                0.8157
## Neg Pred Value                      0.9266                0.9443
## Prevalence                          0.1727                0.1653
## Detection Rate                      0.1125                0.1176
## Detection Prevalence                0.1805                0.1442
## Balanced Accuracy                   0.7847                0.8399
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.43443        0.45980
## Specificity                        0.94802        0.94771
## Pos Pred Value                     0.58242        0.61824
## Neg Pred Value                     0.90945        0.90499
## Prevalence                         0.14302        0.15553
## Detection Rate                     0.06213        0.07151
## Detection Prevalence               0.10668        0.11567
## Balanced Accuracy                  0.69122        0.70375
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(svmModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            150             43             26             31
##   Running 3 METs    54             91             16              2
##   Running 5 METs    15             13             91             17
##   Running 7 METs     5              4             11            120
##   Self Pace walk    19             20              6              1
##   Sitting            8              7             11             19
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      31      48
##   Running 3 METs             35       9
##   Running 5 METs             17      23
##   Running 7 METs              4      12
##   Self Pace walk             66      15
##   Sitting                    10      47
## 
## Overall Statistics
##                                         
##                Accuracy : 0.515         
##                  95% CI : (0.485, 0.545)
##     No Information Rate : 0.2288        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.4103        
##                                         
##  Mcnemar's Test P-Value : 5.882e-10     
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.5976               0.51124
## Specificity                0.7884               0.87378
## Pos Pred Value             0.4559               0.43961
## Neg Pred Value             0.8685               0.90225
## Prevalence                 0.2288               0.16226
## Detection Rate             0.1367               0.08295
## Detection Prevalence       0.2999               0.18870
## Balanced Accuracy          0.6930               0.69251
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                        0.56522                0.6316
## Specificity                        0.90919                0.9603
## Pos Pred Value                     0.51705                0.7692
## Neg Pred Value                     0.92400                0.9256
## Prevalence                         0.14676                0.1732
## Detection Rate                     0.08295                0.1094
## Detection Prevalence               0.16044                0.1422
## Balanced Accuracy                  0.73720                0.7959
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.40491        0.30519
## Specificity                        0.93469        0.94168
## Pos Pred Value                     0.51969        0.46078
## Neg Pred Value                     0.90000        0.89246
## Prevalence                         0.14859        0.14038
## Detection Rate                     0.06016        0.04284
## Detection Prevalence               0.11577        0.09298
## Balanced Accuracy                  0.66980        0.62344
```


##### Fitbit with interpolated data - Random Forest and SVM

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


print(
    "--------------------------------------------Random Forest--------------------------------------------"
  )
```

```
## [1] "--------------------------------------------Random Forest--------------------------------------------"
```

```r
  rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 500)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            358              9              4              7
##   Running 3 METs    15            252              4              2
##   Running 5 METs     5              4            243              2
##   Running 7 METs    10              2              4            336
##   Self Pace walk    12              2              2              2
##   Sitting            8              0             11              8
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       8      12
##   Running 3 METs              2       0
##   Running 5 METs              0      13
##   Running 7 METs              1       8
##   Self Pace walk            233       5
##   Sitting                     9     232
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9063         
##                  95% CI : (0.892, 0.9193)
##     No Information Rate : 0.2236         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.8868         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8775                0.9368
## Specificity                0.9718                0.9852
## Pos Pred Value             0.8995                0.9164
## Neg Pred Value             0.9650                0.9890
## Prevalence                 0.2236                0.1474
## Detection Rate             0.1962                0.1381
## Detection Prevalence       0.2181                0.1507
## Balanced Accuracy          0.9246                0.9610
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.9067                0.9412
## Specificity                         0.9846                0.9830
## Pos Pred Value                      0.9101                0.9307
## Neg Pred Value                      0.9840                0.9857
## Prevalence                          0.1468                0.1956
## Detection Rate                      0.1332                0.1841
## Detection Prevalence                0.1463                0.1978
## Balanced Accuracy                   0.9457                0.9621
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9209         0.8593
## Specificity                         0.9854         0.9768
## Pos Pred Value                      0.9102         0.8657
## Neg Pred Value                      0.9873         0.9756
## Prevalence                          0.1386         0.1479
## Detection Rate                      0.1277         0.1271
## Detection Prevalence                0.1403         0.1468
## Balanced Accuracy                   0.9532         0.9181
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(rfModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            153             12              5              0
##   Running 3 METs    12             97              3              1
##   Running 5 METs     3              0            109              2
##   Running 7 METs     3              0              5            136
##   Self Pace walk     7              0              2              0
##   Sitting            6              0              7              5
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       6       2
##   Running 3 METs              0       0
##   Running 5 METs              1       5
##   Running 7 METs              1       2
##   Self Pace walk             97       2
##   Sitting                     2      97
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8799          
##                  95% CI : (0.8551, 0.9019)
##     No Information Rate : 0.235           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8548          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8315                0.8899
## Specificity                0.9583                0.9763
## Pos Pred Value             0.8596                0.8584
## Neg Pred Value             0.9488                0.9821
## Prevalence                 0.2350                0.1392
## Detection Rate             0.1954                0.1239
## Detection Prevalence       0.2273                0.1443
## Balanced Accuracy          0.8949                0.9331
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.8321                0.9444
## Specificity                         0.9831                0.9828
## Pos Pred Value                      0.9083                0.9252
## Neg Pred Value                      0.9668                0.9874
## Prevalence                          0.1673                0.1839
## Detection Rate                      0.1392                0.1737
## Detection Prevalence                0.1533                0.1877
## Balanced Accuracy                   0.9076                0.9636
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9065         0.8981
## Specificity                         0.9837         0.9704
## Pos Pred Value                      0.8981         0.8291
## Neg Pred Value                      0.9852         0.9835
## Prevalence                          0.1367         0.1379
## Detection Rate                      0.1239         0.1239
## Detection Prevalence                0.1379         0.1494
## Balanced Accuracy                   0.9451         0.9343
```

```r
  print(
    "------------------------------------------------SVM--------------------------------------------------"
  )
```

```
## [1] "------------------------------------------------SVM--------------------------------------------------"
```

```r
  svmModel <- svm(xTrain, yTrain)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            341             30             21             77
##   Running 3 METs    33            187             69             28
##   Running 5 METs     6             40            162             41
##   Running 7 METs    20              0             13            204
##   Self Pace walk     5             12              1              1
##   Sitting            3              0              2              6
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      32     165
##   Running 3 METs            109      35
##   Running 5 METs             28      17
##   Running 7 METs              3      26
##   Self Pace walk             78       7
##   Sitting                     3      20
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5436          
##                  95% CI : (0.5204, 0.5666)
##     No Information Rate : 0.2236          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4423          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8358                0.6952
## Specificity                0.7706                0.8239
## Pos Pred Value             0.5120                0.4056
## Neg Pred Value             0.9422                0.9399
## Prevalence                 0.2236                0.1474
## Detection Rate             0.1868                0.1025
## Detection Prevalence       0.3649                0.2526
## Balanced Accuracy          0.8032                0.7595
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                        0.60448                0.5714
## Specificity                        0.91522                0.9578
## Pos Pred Value                     0.55102                0.7669
## Neg Pred Value                     0.93076                0.9019
## Prevalence                         0.14685                0.1956
## Detection Rate                     0.08877                0.1118
## Detection Prevalence               0.16110                0.1458
## Balanced Accuracy                  0.75985                0.7646
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.30830        0.07407
## Specificity                        0.98346        0.99100
## Pos Pred Value                     0.75000        0.58824
## Neg Pred Value                     0.89831        0.86041
## Prevalence                         0.13863        0.14795
## Detection Rate                     0.04274        0.01096
## Detection Prevalence               0.05699        0.01863
## Balanced Accuracy                  0.64588        0.53254
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(svmModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            142             18             21             30
##   Running 3 METs    20             57             44              8
##   Running 5 METs     7             17             52             12
##   Running 7 METs    11              1              7             88
##   Self Pace walk     2             16              3              3
##   Sitting            2              0              4              3
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      19      68
##   Running 3 METs             40      11
##   Running 5 METs             13       5
##   Running 7 METs              5      17
##   Self Pace walk             29       2
##   Sitting                     1       5
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4764         
##                  95% CI : (0.4409, 0.512)
##     No Information Rate : 0.235          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3562         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.7717                0.5229
## Specificity                0.7396                0.8175
## Pos Pred Value             0.4765                0.3167
## Neg Pred Value             0.9134                0.9138
## Prevalence                 0.2350                0.1392
## Detection Rate             0.1814                0.0728
## Detection Prevalence       0.3806                0.2299
## Balanced Accuracy          0.7557                0.6702
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                        0.39695                0.6111
## Specificity                        0.91718                0.9358
## Pos Pred Value                     0.49057                0.6822
## Neg Pred Value                     0.88331                0.9144
## Prevalence                         0.16731                0.1839
## Detection Rate                     0.06641                0.1124
## Detection Prevalence               0.13538                0.1648
## Balanced Accuracy                  0.65706                0.7735
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.27103       0.046296
## Specificity                        0.96154       0.985185
## Pos Pred Value                     0.52727       0.333333
## Neg Pred Value                     0.89286       0.865885
## Prevalence                         0.13665       0.137931
## Detection Rate                     0.03704       0.006386
## Detection Prevalence               0.07024       0.019157
## Balanced Accuracy                  0.61628       0.515741
```


#### AppleWatch with interpolated data - Rotation Forest

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
```

```
## ```

```r
# load the RotationForest classifier using RWeka
rotation_forest <- make_Weka_classifier("weka/classifiers/meta/RotationForest")

# generate the RotationForest model
rotationForestModel <- rotation_forest(activity_trimmed ~ ., train)

# predict the results
rotationForestPredict <- predict(rotationForestModel, test, type = 'class')

print(confusionMatrix(rotationForestPredict, test$activity_trimmed))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            180             22              4              7
##   Running 3 METs    17            135              4              0
##   Running 5 METs     8              7            152              5
##   Running 7 METs     3              0              9            147
##   Self Pace walk    13              3              1              0
##   Sitting           17              5              7              4
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      11      26
##   Running 3 METs              6       2
##   Running 5 METs              1      12
##   Running 7 METs              0      12
##   Self Pace walk            148      14
##   Sitting                     6     111
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7944          
##                  95% CI : (0.7692, 0.8179)
##     No Information Rate : 0.2166          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7522          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.7563                0.7849
## Specificity                0.9187                0.9687
## Pos Pred Value             0.7200                0.8232
## Neg Pred Value             0.9317                0.9604
## Prevalence                 0.2166                0.1565
## Detection Rate             0.1638                0.1228
## Detection Prevalence       0.2275                0.1492
## Balanced Accuracy          0.8375                0.8768
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.8588                0.9018
## Specificity                         0.9642                0.9744
## Pos Pred Value                      0.8216                0.8596
## Neg Pred Value                      0.9726                0.9828
## Prevalence                          0.1611                0.1483
## Detection Rate                      0.1383                0.1338
## Detection Prevalence                0.1683                0.1556
## Balanced Accuracy                   0.9115                0.9381
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8605         0.6271
## Specificity                         0.9666         0.9577
## Pos Pred Value                      0.8268         0.7400
## Neg Pred Value                      0.9739         0.9305
## Prevalence                          0.1565         0.1611
## Detection Rate                      0.1347         0.1010
## Detection Prevalence                0.1629         0.1365
## Balanced Accuracy                   0.9135         0.7924
```


#### Fitbit with interpolated data - Rotation Forest

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
```

```
## ```

```r
# load the RotationForest classifier using RWeka
rotation_forest <- make_Weka_classifier("weka/classifiers/meta/RotationForest")

# generate the RotationForest model
rotationForestModel <- rotation_forest(activity_trimmed ~ ., train)

# predict the results
rotationForestPredict <- predict(rotationForestModel, test, type = 'class')

print(confusionMatrix(rotationForestPredict, test$activity_trimmed))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            149              6              6              2
##   Running 3 METs     7            102              3              0
##   Running 5 METs     2              2            116              2
##   Running 7 METs     0              1              1            144
##   Self Pace walk     7              0              0              4
##   Sitting            4              0              1              8
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       2       3
##   Running 3 METs              1       0
##   Running 5 METs              1       6
##   Running 7 METs              0       6
##   Self Pace walk             94       1
##   Sitting                     3     109
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9004          
##                  95% CI : (0.8774, 0.9203)
##     No Information Rate : 0.2131          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8797          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8817                0.9189
## Specificity                0.9696                0.9839
## Pos Pred Value             0.8869                0.9027
## Neg Pred Value             0.9680                0.9868
## Prevalence                 0.2131                0.1400
## Detection Rate             0.1879                0.1286
## Detection Prevalence       0.2119                0.1425
## Balanced Accuracy          0.9256                0.9514
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.9134                0.9000
## Specificity                         0.9805                0.9874
## Pos Pred Value                      0.8992                0.9474
## Neg Pred Value                      0.9834                0.9750
## Prevalence                          0.1602                0.2018
## Detection Rate                      0.1463                0.1816
## Detection Prevalence                0.1627                0.1917
## Balanced Accuracy                   0.9469                0.9437
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9307         0.8720
## Specificity                         0.9827         0.9760
## Pos Pred Value                      0.8868         0.8720
## Neg Pred Value                      0.9898         0.9760
## Prevalence                          0.1274         0.1576
## Detection Rate                      0.1185         0.1375
## Detection Prevalence                0.1337         0.1576
## Balanced Accuracy                   0.9567         0.9240
```



#### Generating AppleWatch and Fitbit data for Rotation Forest model in Weka

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


##### Fitbit with interpolated data, removing Nas with rfImpute - Random Forest and SVM

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
  "FitbitStepsXDistance"
  ,"activity_trimmed"
)

summary_data_before_removing_na <- aggregated_data %>% 
  summarise(
    total_row_num = dim(aggregated_data)[1],
    n_Fitbit.Steps = length(which(!is.na(Fitbit.Steps))),
    n_Fitbit.Heart = length(which(!is.na(Fitbit.Heart))),
    n_Fitbit.Calories = length(which(!is.na(Fitbit.Calories))),
    n_FFitbit.Distance = length(which(!is.na(Fitbit.Distance))),
    n_EntropyFitbitHeartPerDay = length(which(!is.na(EntropyFitbitHeartPerDay))),
    n_EntropyFitbitStepsPerDay = length(which(!is.na(EntropyFitbitStepsPerDay))),
    n_RestingFitbitHeartrate = length(which(!is.na(RestingFitbitHeartrate))),
    n_CorrelationFitbitHeartrateSteps = length(which(!is.na(CorrelationFitbitHeartrateSteps))),
    n_NormalizedFitbitHeartrate = length(which(!is.na(NormalizedFitbitHeartrate))),
    n_FitbitIntensity = length(which(!is.na(FitbitIntensity))),
    n_SDNormalizedFitbitHR = length(which(!is.na(SDNormalizedFitbitHR))),
    n_FitbitStepsXDistance = length(which(!is.na(FitbitStepsXDistance)))
  )

aggregated_data_impute <- aggregated_data %>% select(x_columns_FB)
aggregated_data_impute$gender <- ifelse(aggregated_data_impute$gender == "Male", 1, 0)


#idx <- which(is.na(x), arr.ind = T)[, 1]
#x <- x[-idx, ]

# write.csv(x, "data_for_process_fb.csv")

aggregated_data_impute$activity_trimmed <- as_factor(aggregated_data_impute$activity_trimmed)
# removing Nas from the dataset
aggregated_data_impute <- as.data.frame(aggregated_data_impute)
aggregated_data_imputed <- rfImpute(activity_trimmed ~ ., aggregated_data_impute)
```

```
## ntree      OOB      1      2      3      4      5      6
##   300:  39.88% 24.40% 70.29% 62.76% 41.64% 22.89% 27.69%
## ntree      OOB      1      2      3      4      5      6
##   300:  40.97% 33.67% 58.51% 57.28% 48.63% 28.52% 25.41%
## ntree      OOB      1      2      3      4      5      6
##   300:  40.81% 30.88% 57.97% 58.41% 47.61% 31.18% 25.90%
## ntree      OOB      1      2      3      4      5      6
##   300:  39.96% 30.11% 57.61% 55.20% 46.08% 31.01% 26.55%
## ntree      OOB      1      2      3      4      5      6
##   300:  40.83% 30.37% 63.77% 56.71% 44.54% 31.34% 25.73%
```

```r
summary_data_after_removing_na <- aggregated_data_imputed %>% 
  summarise(
    total_row_num = dim(aggregated_data_imputed)[1],
    n_Fitbit.Steps_LE = length(which(!is.na(Fitbit.Steps))),
    n_Fitbit.Heart_LE = length(which(!is.na(Fitbit.Heart))),
    n_Fitbit.Calories_LE = length(which(!is.na(Fitbit.Calories))),
    n_FFitbit.Distance_LE = length(which(!is.na(Fitbit.Distance))),
    n_EntropyFitbitHeartPerDay_LE = length(which(!is.na(EntropyFitbitHeartPerDay))),
    n_EntropyFitbitStepsPerDay_LE = length(which(!is.na(EntropyFitbitStepsPerDay))),
    n_RestingFitbitHeartrate_LE = length(which(!is.na(RestingFitbitHeartrate))),
    n_CorrelationFitbitHeartrateSteps_LE = length(which(!is.na(CorrelationFitbitHeartrateSteps))),
    n_NormalizedFitbitHeartrate_LE = length(which(!is.na(NormalizedFitbitHeartrate))),
    n_FitbitIntensity_LE = length(which(!is.na(FitbitIntensity))),
    n_SDNormalizedFitbitHR_LE = length(which(!is.na(SDNormalizedFitbitHR))),
    n_FitbitStepsXDistance_LE = length(which(!is.na(FitbitStepsXDistance)))
  )


#y <- as.factor(aggregated_data$activity_trimmed[-idx])
y <- as.factor(aggregated_data_imputed$activity_trimmed)
x <- select(aggregated_data_imputed,-c(1))


train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

xTrain = x[train, ]
yTrain = y[train]
xTest = x[test, ]
yTest = y[test]




print(
    "--------------------------------------------Random Forest--------------------------------------------"
  )
```

```
## [1] "--------------------------------------------Random Forest--------------------------------------------"
```

```r
  rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 500)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Sitting Self Pace walk Running 3 METs
##   Lying            370     152             23             16
##   Sitting           99     143             23             21
##   Self Pace walk    34      32            143             92
##   Running 3 METs    24      26            113            213
##   Running 5 METs     8      26             48             67
##   Running 7 METs    19       9             21             11
##                 Reference
## Prediction       Running 5 METs Running 7 METs
##   Lying                      14             18
##   Sitting                    12             17
##   Self Pace walk             32             20
##   Running 3 METs             62             21
##   Running 5 METs            266             47
##   Running 7 METs             20            307
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5613          
##                  95% CI : (0.5419, 0.5806)
##     No Information Rate : 0.2156          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4708          
##                                           
##  Mcnemar's Test P-Value : 5.77e-05        
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Sitting Class: Self Pace walk
## Sensitivity                0.6679        0.36856               0.38544
## Specificity                0.8893        0.92114               0.90446
## Pos Pred Value             0.6239        0.45397               0.40510
## Neg Pred Value             0.9069        0.89130               0.89711
## Prevalence                 0.2156        0.15103               0.14441
## Detection Rate             0.1440        0.05566               0.05566
## Detection Prevalence       0.2308        0.12262               0.13741
## Balanced Accuracy          0.7786        0.64485               0.64495
##                      Class: Running 3 METs Class: Running 5 METs
## Sensitivity                        0.50714                0.6552
## Specificity                        0.88553                0.9094
## Pos Pred Value                     0.46405                0.5758
## Neg Pred Value                     0.90190                0.9336
## Prevalence                         0.16349                0.1580
## Detection Rate                     0.08291                0.1035
## Detection Prevalence               0.17867                0.1798
## Balanced Accuracy                  0.69634                0.7823
##                      Class: Running 7 METs
## Sensitivity                         0.7140
## Specificity                         0.9626
## Pos Pred Value                      0.7933
## Neg Pred Value                      0.9436
## Prevalence                          0.1674
## Detection Rate                      0.1195
## Detection Prevalence                0.1506
## Balanced Accuracy                   0.8383
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(rfModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Sitting Self Pace walk Running 3 METs
##   Lying            163      55             10              6
##   Sitting           38      64             12              6
##   Self Pace walk    15      15             58             44
##   Running 3 METs    13      11             53             86
##   Running 5 METs     1      11             21             19
##   Running 7 METs     3       8              4              5
##                 Reference
## Prediction       Running 5 METs Running 7 METs
##   Lying                       7              4
##   Sitting                     8              5
##   Self Pace walk             20              6
##   Running 3 METs             35              5
##   Running 5 METs            119             22
##   Running 7 METs              8            142
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5735          
##                  95% CI : (0.5437, 0.6029)
##     No Information Rate : 0.2114          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.4859          
##                                           
##  Mcnemar's Test P-Value : 0.03013         
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Sitting Class: Self Pace walk
## Sensitivity                0.6996        0.39024               0.36709
## Specificity                0.9056        0.92644               0.89407
## Pos Pred Value             0.6653        0.48120               0.36709
## Neg Pred Value             0.9183        0.89680               0.89407
## Prevalence                 0.2114        0.14882               0.14338
## Detection Rate             0.1479        0.05808               0.05263
## Detection Prevalence       0.2223        0.12069               0.14338
## Balanced Accuracy          0.8026        0.65834               0.63058
##                      Class: Running 3 METs Class: Running 5 METs
## Sensitivity                        0.51807                0.6041
## Specificity                        0.87500                0.9182
## Pos Pred Value                     0.42365                0.6166
## Neg Pred Value                     0.91101                0.9142
## Prevalence                         0.15064                0.1788
## Detection Rate                     0.07804                0.1080
## Detection Prevalence               0.18421                0.1751
## Balanced Accuracy                  0.69654                0.7611
##                      Class: Running 7 METs
## Sensitivity                         0.7717
## Specificity                         0.9695
## Pos Pred Value                      0.8353
## Neg Pred Value                      0.9549
## Prevalence                          0.1670
## Detection Rate                      0.1289
## Detection Prevalence                0.1543
## Balanced Accuracy                   0.8706
```

```r
  print(
    "------------------------------------------------SVM--------------------------------------------------"
  )
```

```
## [1] "------------------------------------------------SVM--------------------------------------------------"
```

```r
  svmModel <- svm(xTrain, yTrain)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Sitting Self Pace walk Running 3 METs
##   Lying            410     161             26             22
##   Sitting           47     132             22             14
##   Self Pace walk    12      11             89             19
##   Running 3 METs    42      43            146            253
##   Running 5 METs    14      24             68             97
##   Running 7 METs    29      17             20             15
##                 Reference
## Prediction       Running 5 METs Running 7 METs
##   Lying                      21             88
##   Sitting                    19             24
##   Self Pace walk              7              8
##   Running 3 METs             80             28
##   Running 5 METs            267             48
##   Running 7 METs             12            234
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5391          
##                  95% CI : (0.5196, 0.5585)
##     No Information Rate : 0.2156          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4411          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Sitting Class: Self Pace walk
## Sensitivity                0.7401        0.34021               0.23989
## Specificity                0.8422        0.94223               0.97407
## Pos Pred Value             0.5632        0.51163               0.60959
## Neg Pred Value             0.9218        0.88923               0.88362
## Prevalence                 0.2156        0.15103               0.14441
## Detection Rate             0.1596        0.05138               0.03464
## Detection Prevalence       0.2834        0.10043               0.05683
## Balanced Accuracy          0.7911        0.64122               0.60698
##                      Class: Running 3 METs Class: Running 5 METs
## Sensitivity                        0.60238                0.6576
## Specificity                        0.84225                0.8840
## Pos Pred Value                     0.42736                0.5154
## Neg Pred Value                     0.91553                0.9322
## Prevalence                         0.16349                0.1580
## Detection Rate                     0.09848                0.1039
## Detection Prevalence               0.23044                0.2016
## Balanced Accuracy                  0.72232                0.7708
##                      Class: Running 7 METs
## Sensitivity                        0.54419
## Specificity                        0.95652
## Pos Pred Value                     0.71560
## Neg Pred Value                     0.91258
## Prevalence                         0.16738
## Detection Rate                     0.09109
## Detection Prevalence               0.12729
## Balanced Accuracy                  0.75035
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(svmModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Sitting Self Pace walk Running 3 METs
##   Lying            164      71             12              9
##   Sitting           26      42             13             12
##   Self Pace walk     5       9             22             15
##   Running 3 METs    19      23             67             89
##   Running 5 METs     4      11             40             33
##   Running 7 METs    15       8              4              8
##                 Reference
## Prediction       Running 5 METs Running 7 METs
##   Lying                      11             35
##   Sitting                    14             11
##   Self Pace walk              6              3
##   Running 3 METs             49              9
##   Running 5 METs            107             36
##   Running 7 METs             10             90
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4664          
##                  95% CI : (0.4366, 0.4964)
##     No Information Rate : 0.2114          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3536          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Sitting Class: Self Pace walk
## Sensitivity                0.7039        0.25610               0.13924
## Specificity                0.8412        0.91898               0.95975
## Pos Pred Value             0.5430        0.35593               0.36667
## Neg Pred Value             0.9138        0.87602               0.86948
## Prevalence                 0.2114        0.14882               0.14338
## Detection Rate             0.1488        0.03811               0.01996
## Detection Prevalence       0.2740        0.10708               0.05445
## Balanced Accuracy          0.7725        0.58754               0.54949
##                      Class: Running 3 METs Class: Running 5 METs
## Sensitivity                        0.53614                0.5431
## Specificity                        0.82158                0.8630
## Pos Pred Value                     0.34766                0.4632
## Neg Pred Value                     0.90898                0.8967
## Prevalence                         0.15064                0.1788
## Detection Rate                     0.08076                0.0971
## Detection Prevalence               0.23230                0.2096
## Balanced Accuracy                  0.67886                0.7031
##                      Class: Running 7 METs
## Sensitivity                        0.48913
## Specificity                        0.95098
## Pos Pred Value                     0.66667
## Neg Pred Value                     0.90279
## Prevalence                         0.16697
## Detection Rate                     0.08167
## Detection Prevalence               0.12250
## Balanced Accuracy                  0.72006
```

##### Applewatch with interpolated data, imputed - Random Forest and SVM

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
  "ApplewatchStepsXDistance"
)

x <- aggregated_data %>% select(x_columns_AW)
x$gender <- ifelse(x$gender == "Male", 1, 0)
x <- impute(x = x, what = "mean")


y <- as.factor(aggregated_data_imputed$activity_trimmed)

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

xTrain = x[train, ]
yTrain = y[train]
xTest = x[test, ]
yTest = y[test]


print(
    "--------------------------------------------Random Forest--------------------------------------------"
  )
```

```
## [1] "--------------------------------------------Random Forest--------------------------------------------"
```

```r
  rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 100)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Sitting Self Pace walk Running 3 METs
##   Lying            275     117             42             54
##   Sitting          106     130             39             39
##   Self Pace walk    42      37            156             64
##   Running 3 METs    64      56             84            189
##   Running 5 METs    31      33             44             50
##   Running 7 METs    27      15              7              6
##                 Reference
## Prediction       Running 5 METs Running 7 METs
##   Lying                      35             56
##   Sitting                    18             16
##   Self Pace walk             42              4
##   Running 3 METs             65             22
##   Running 5 METs            237             36
##   Running 7 METs             26            305
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5029          
##                  95% CI : (0.4834, 0.5224)
##     No Information Rate : 0.2121          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4009          
##                                           
##  Mcnemar's Test P-Value : 0.002029        
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Sitting Class: Self Pace walk
## Sensitivity                0.5046         0.3351               0.41935
## Specificity                0.8498         0.9000               0.91397
## Pos Pred Value             0.4750         0.3736               0.45217
## Neg Pred Value             0.8643         0.8838               0.90288
## Prevalence                 0.2121         0.1510               0.14480
## Detection Rate             0.1070         0.0506               0.06072
## Detection Prevalence       0.2254         0.1355               0.13429
## Balanced Accuracy          0.6772         0.6175               0.66666
##                      Class: Running 3 METs Class: Running 5 METs
## Sensitivity                        0.47015               0.56028
## Specificity                        0.86571               0.90960
## Pos Pred Value                     0.39375               0.54988
## Neg Pred Value                     0.89804               0.91300
## Prevalence                         0.15648               0.16466
## Detection Rate                     0.07357               0.09225
## Detection Prevalence               0.18684               0.16777
## Balanced Accuracy                  0.66793               0.73494
##                      Class: Running 7 METs
## Sensitivity                         0.6948
## Specificity                         0.9620
## Pos Pred Value                      0.7902
## Neg Pred Value                      0.9386
## Prevalence                          0.1709
## Detection Rate                      0.1187
## Detection Prevalence                0.1503
## Balanced Accuracy                   0.8284
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(rfModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Sitting Self Pace walk Running 3 METs
##   Lying            127      44             20             38
##   Sitting           35      62              9              9
##   Self Pace walk    17      16             57             26
##   Running 3 METs    35      19             53             87
##   Running 5 METs    15      15             13             21
##   Running 7 METs    13       8              5              3
##                 Reference
## Prediction       Running 5 METs Running 7 METs
##   Lying                      17             22
##   Sitting                    10              5
##   Self Pace walk              7              5
##   Running 3 METs             27             13
##   Running 5 METs            100             14
##   Running 7 METs             19            116
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4982          
##                  95% CI : (0.4682, 0.5281)
##     No Information Rate : 0.2196          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.3932          
##                                           
##  Mcnemar's Test P-Value : 0.01251         
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Sitting Class: Self Pace walk
## Sensitivity                0.5248        0.37805               0.36306
## Specificity                0.8360        0.92751               0.92487
## Pos Pred Value             0.4739        0.47692               0.44531
## Neg Pred Value             0.8621        0.89506               0.89733
## Prevalence                 0.2196        0.14882               0.14247
## Detection Rate             0.1152        0.05626               0.05172
## Detection Prevalence       0.2432        0.11797               0.11615
## Balanced Accuracy          0.6804        0.65278               0.64396
##                      Class: Running 3 METs Class: Running 5 METs
## Sensitivity                        0.47283               0.55556
## Specificity                        0.83987               0.91540
## Pos Pred Value                     0.37179               0.56180
## Neg Pred Value                     0.88825               0.91342
## Prevalence                         0.16697               0.16334
## Detection Rate                     0.07895               0.09074
## Detection Prevalence               0.21234               0.16152
## Balanced Accuracy                  0.65635               0.73548
##                      Class: Running 7 METs
## Sensitivity                         0.6629
## Specificity                         0.9482
## Pos Pred Value                      0.7073
## Neg Pred Value                      0.9371
## Prevalence                          0.1588
## Detection Rate                      0.1053
## Detection Prevalence                0.1488
## Balanced Accuracy                   0.8055
```

```r
  print(
    "------------------------------------------------SVM--------------------------------------------------"
  )
```

```
## [1] "------------------------------------------------SVM--------------------------------------------------"
```

```r
  svmModel <- svm(xTrain, yTrain)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Sitting Self Pace walk Running 3 METs
##   Lying            509     320            178            222
##   Sitting            4      19              7              2
##   Self Pace walk    13      10             64             33
##   Running 3 METs    10       7             95            107
##   Running 5 METs     8      17             25             33
##   Running 7 METs     1      15              3              5
##                 Reference
## Prediction       Running 5 METs Running 7 METs
##   Lying                     179            168
##   Sitting                     3              2
##   Self Pace walk             30              3
##   Running 3 METs             62             17
##   Running 5 METs            133             33
##   Running 7 METs             16            216
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4079          
##                  95% CI : (0.3889, 0.4272)
##     No Information Rate : 0.2121          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.2672          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Sitting Class: Self Pace walk
## Sensitivity                0.9339       0.048969               0.17204
## Specificity                0.4728       0.991747               0.95949
## Pos Pred Value             0.3230       0.513514               0.41830
## Neg Pred Value             0.9637       0.854265               0.87252
## Prevalence                 0.2121       0.151032               0.14480
## Detection Rate             0.1981       0.007396               0.02491
## Detection Prevalence       0.6135       0.014402               0.05956
## Balanced Accuracy          0.7034       0.520358               0.56577
##                      Class: Running 3 METs Class: Running 5 METs
## Sensitivity                        0.26617               0.31442
## Specificity                        0.91186               0.94595
## Pos Pred Value                     0.35906               0.53414
## Neg Pred Value                     0.87010               0.87500
## Prevalence                         0.15648               0.16466
## Detection Rate                     0.04165               0.05177
## Detection Prevalence               0.11600               0.09692
## Balanced Accuracy                  0.58901               0.63018
##                      Class: Running 7 METs
## Sensitivity                        0.49203
## Specificity                        0.98122
## Pos Pred Value                     0.84375
## Neg Pred Value                     0.90359
## Prevalence                         0.17088
## Detection Rate                     0.08408
## Detection Prevalence               0.09965
## Balanced Accuracy                  0.73662
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(svmModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Sitting Self Pace walk Running 3 METs
##   Lying            227     136             80            111
##   Sitting            1       8              3              3
##   Self Pace walk     2       3             19             15
##   Running 3 METs     8       3             41             34
##   Running 5 METs     2       3              7             18
##   Running 7 METs     2      11              7              3
##                 Reference
## Prediction       Running 5 METs Running 7 METs
##   Lying                      80             65
##   Sitting                     4              3
##   Self Pace walk             11              3
##   Running 3 METs             29              9
##   Running 5 METs             37             24
##   Running 7 METs             19             71
## 
## Overall Statistics
##                                          
##                Accuracy : 0.3593         
##                  95% CI : (0.331, 0.3885)
##     No Information Rate : 0.2196         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.2015         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Sitting Class: Self Pace walk
## Sensitivity                0.9380        0.04878               0.12102
## Specificity                0.4512        0.98507               0.96402
## Pos Pred Value             0.3247        0.36364               0.35849
## Neg Pred Value             0.9628        0.85556               0.86845
## Prevalence                 0.2196        0.14882               0.14247
## Detection Rate             0.2060        0.00726               0.01724
## Detection Prevalence       0.6343        0.01996               0.04809
## Balanced Accuracy          0.6946        0.51693               0.54252
##                      Class: Running 3 METs Class: Running 5 METs
## Sensitivity                        0.18478               0.20556
## Specificity                        0.90196               0.94143
## Pos Pred Value                     0.27419               0.40659
## Neg Pred Value                     0.84663               0.85856
## Prevalence                         0.16697               0.16334
## Detection Rate                     0.03085               0.03358
## Detection Prevalence               0.11252               0.08258
## Balanced Accuracy                  0.54337               0.57349
##                      Class: Running 7 METs
## Sensitivity                        0.40571
## Specificity                        0.95469
## Pos Pred Value                     0.62832
## Neg Pred Value                     0.89484
## Prevalence                         0.15880
## Detection Rate                     0.06443
## Detection Prevalence               0.10254
## Balanced Accuracy                  0.68020
```

##### Fitbit with non-interpolated data, imputed - Random Forest and SVM

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
  "FitbitStepsXDistance"
)

summary_data_before_imputation <- aggregated_data %>% 
  summarise(
    total_row_num = dim(aggregated_data)[1],
    n_Fitbit.Steps = length(which(!is.na(Fitbit.Steps))),
    n_Fitbit.Heart = length(which(!is.na(Fitbit.Heart))),
    n_Fitbit.Calories = length(which(!is.na(Fitbit.Calories))),
    n_FFitbit.Distance = length(which(!is.na(Fitbit.Distance))),
    n_EntropyFitbitHeartPerDay = length(which(!is.na(EntropyFitbitHeartPerDay))),
    n_EntropyFitbitStepsPerDay = length(which(!is.na(EntropyFitbitStepsPerDay))),
    n_RestingFitbitHeartrate = length(which(!is.na(RestingFitbitHeartrate))),
    n_CorrelationFitbitHeartrateSteps = length(which(!is.na(CorrelationFitbitHeartrateSteps))),
    n_NormalizedFitbitHeartrate = length(which(!is.na(NormalizedFitbitHeartrate))),
    n_FitbitIntensity = length(which(!is.na(FitbitIntensity))),
    n_SDNormalizedFitbitHR = length(which(!is.na(SDNormalizedFitbitHR))),
    n_FitbitStepsXDistance = length(which(!is.na(FitbitStepsXDistance)))
  )

x <- aggregated_data %>% select(x_columns_FB)
x$gender <- ifelse(x$gender == "Male", 1, 0)
x <- as.data.frame(impute(x = x, what = "mean"))

summary_data_after_imputation <- x %>% 
 summarise(
    total_row_num = dim(x)[1],
    n_Fitbit.Steps = length(which(!is.na(Fitbit.Steps))),
    n_Fitbit.Heart = length(which(!is.na(Fitbit.Heart))),
    n_Fitbit.Calories = length(which(!is.na(Fitbit.Calories))),
    n_FFitbit.Distance = length(which(!is.na(Fitbit.Distance))),
    n_EntropyFitbitHeartPerDay = length(which(!is.na(EntropyFitbitHeartPerDay))),
    n_EntropyFitbitStepsPerDay = length(which(!is.na(EntropyFitbitStepsPerDay))),
    n_RestingFitbitHeartrate = length(which(!is.na(RestingFitbitHeartrate))),
    n_CorrelationFitbitHeartrateSteps = length(which(!is.na(CorrelationFitbitHeartrateSteps))),
    n_NormalizedFitbitHeartrate = length(which(!is.na(NormalizedFitbitHeartrate))),
    n_FitbitIntensity = length(which(!is.na(FitbitIntensity))),
    n_SDNormalizedFitbitHR = length(which(!is.na(SDNormalizedFitbitHR))),
    n_FitbitStepsXDistance = length(which(!is.na(FitbitStepsXDistance)))
  )


y <- as.factor(aggregated_data$activity_trimmed)

train = sample(seq(length(y)), 0.7 * length(y))
test = seq(length(y))[-train]

xTrain = x[train, ]
yTrain = y[train]
xTest = x[test, ]
yTest = y[test]

print(
    "--------------------------------------------Random Forest--------------------------------------------"
  )
```

```
## [1] "--------------------------------------------Random Forest--------------------------------------------"
```

```r
  rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 100)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            384             31             25             20
##   Running 3 METs    37            222             48             21
##   Running 5 METs     8             55            315             48
##   Running 7 METs    18              9              9            316
##   Self Pace walk    34             81             24             10
##   Sitting           70             11             12             10
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      21     136
##   Running 3 METs            112      23
##   Running 5 METs             52      26
##   Running 7 METs              7       7
##   Self Pace walk            163      29
##   Sitting                    13     162
## 
## Overall Statistics
##                                          
##                Accuracy : 0.608          
##                  95% CI : (0.5888, 0.627)
##     No Information Rate : 0.2145         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.5266         
##                                          
##  Mcnemar's Test P-Value : 4.027e-14      
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.6969               0.54279
## Specificity                0.8845               0.88843
## Pos Pred Value             0.6224               0.47948
## Neg Pred Value             0.9144               0.91121
## Prevalence                 0.2145               0.15921
## Detection Rate             0.1495               0.08641
## Detection Prevalence       0.2402               0.18023
## Balanced Accuracy          0.7907               0.71561
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.7275                0.7435
## Specificity                         0.9115                0.9767
## Pos Pred Value                      0.6250                0.8634
## Neg Pred Value                      0.9429                0.9505
## Prevalence                          0.1685                0.1654
## Detection Rate                      0.1226                0.1230
## Detection Prevalence                0.1962                0.1425
## Balanced Accuracy                   0.8195                0.8601
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.44293        0.42298
## Specificity                        0.91913        0.94694
## Pos Pred Value                     0.47801        0.58273
## Neg Pred Value                     0.90799        0.90354
## Prevalence                         0.14325        0.14909
## Detection Rate                     0.06345        0.06306
## Detection Prevalence               0.13274        0.10821
## Balanced Accuracy                  0.68103        0.68496
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(rfModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            165             11             12             14
##   Running 3 METs    16            111             25              6
##   Running 5 METs     1             11            119             20
##   Running 7 METs     5              1              6            137
##   Self Pace walk    16             39              6              8
##   Sitting           33              4              2              4
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      15      66
##   Running 3 METs             36      17
##   Running 5 METs             20      13
##   Running 7 METs              5       3
##   Self Pace walk             71      20
##   Sitting                    14      50
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5926          
##                  95% CI : (0.5629, 0.6217)
##     No Information Rate : 0.2142          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5079          
##                                           
##  Mcnemar's Test P-Value : 1.12e-08        
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.6992                0.6271
## Specificity                0.8637                0.8919
## Pos Pred Value             0.5830                0.5261
## Neg Pred Value             0.9133                0.9259
## Prevalence                 0.2142                0.1606
## Detection Rate             0.1497                0.1007
## Detection Prevalence       0.2568                0.1915
## Balanced Accuracy          0.7814                0.7595
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.7000                0.7249
## Specificity                         0.9303                0.9781
## Pos Pred Value                      0.6467                0.8726
## Neg Pred Value                      0.9444                0.9450
## Prevalence                          0.1543                0.1715
## Detection Rate                      0.1080                0.1243
## Detection Prevalence                0.1670                0.1425
## Balanced Accuracy                   0.8151                0.8515
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.44099        0.29586
## Specificity                        0.90542        0.93891
## Pos Pred Value                     0.44375        0.46729
## Neg Pred Value                     0.90446        0.88040
## Prevalence                         0.14610        0.15336
## Detection Rate                     0.06443        0.04537
## Detection Prevalence               0.14519        0.09710
## Balanced Accuracy                  0.67321        0.61738
```

```r
  print(
    "------------------------------------------------SVM--------------------------------------------------"
  )
```

```
## [1] "------------------------------------------------SVM--------------------------------------------------"
```

```r
  svmModel <- svm(xTrain, yTrain)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            444             38             41            142
##   Running 3 METs    48            185             60             23
##   Running 5 METs    19            163            310             58
##   Running 7 METs    24              3              3            181
##   Self Pace walk    10             20             13              7
##   Sitting            6              0              6             14
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      43     244
##   Running 3 METs            107      36
##   Running 5 METs            135      37
##   Running 7 METs              5       7
##   Self Pace walk             75      13
##   Sitting                     3      46
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4831          
##                  95% CI : (0.4636, 0.5026)
##     No Information Rate : 0.2145          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3683          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8058               0.45232
## Specificity                0.7483               0.87315
## Pos Pred Value             0.4664               0.40305
## Neg Pred Value             0.9338               0.89384
## Prevalence                 0.2145               0.15921
## Detection Rate             0.1728               0.07201
## Detection Prevalence       0.3706               0.17867
## Balanced Accuracy          0.7770               0.66274
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.7159               0.42588
## Specificity                         0.8071               0.98041
## Pos Pred Value                      0.4294               0.81166
## Neg Pred Value                      0.9334               0.89599
## Prevalence                          0.1685               0.16543
## Detection Rate                      0.1207               0.07046
## Detection Prevalence                0.2810               0.08680
## Balanced Accuracy                   0.7615               0.70315
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.20380        0.12010
## Specificity                        0.97138        0.98673
## Pos Pred Value                     0.54348        0.61333
## Neg Pred Value                     0.87947        0.86488
## Prevalence                         0.14325        0.14909
## Detection Rate                     0.02919        0.01791
## Detection Prevalence               0.05372        0.02919
## Balanced Accuracy                  0.58759        0.55342
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(svmModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            181             15             17             75
##   Running 3 METs    21             65             39              9
##   Running 5 METs     9             79            101             19
##   Running 7 METs    11              2              4             70
##   Self Pace walk     4             15              8              4
##   Sitting           10              1              1             12
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      31      93
##   Running 3 METs             54      31
##   Running 5 METs             44      14
##   Running 7 METs              7       8
##   Self Pace walk             22       8
##   Sitting                     3      15
## 
## Overall Statistics
##                                           
##                Accuracy : 0.412           
##                  95% CI : (0.3827, 0.4417)
##     No Information Rate : 0.2142          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.2837          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.7669               0.36723
## Specificity                0.7333               0.83351
## Pos Pred Value             0.4393               0.29680
## Neg Pred Value             0.9203               0.87316
## Prevalence                 0.2142               0.16062
## Detection Rate             0.1642               0.05898
## Detection Prevalence       0.3739               0.19873
## Balanced Accuracy          0.7501               0.60037
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                        0.59412               0.37037
## Specificity                        0.82296               0.96495
## Pos Pred Value                     0.37970               0.68627
## Neg Pred Value                     0.91746               0.88100
## Prevalence                         0.15426               0.17151
## Detection Rate                     0.09165               0.06352
## Detection Prevalence               0.24138               0.09256
## Balanced Accuracy                  0.70854               0.66766
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.13665        0.08876
## Specificity                        0.95855        0.97106
## Pos Pred Value                     0.36066        0.35714
## Neg Pred Value                     0.86647        0.85472
## Prevalence                         0.14610        0.15336
## Detection Rate                     0.01996        0.01361
## Detection Prevalence               0.05535        0.03811
## Balanced Accuracy                  0.54760        0.52991
```


##### predicting EE by linear regression

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
  "EE"
)


x <- aggregated_data %>% select(x_columns_AW) %>% 
  mutate(gender = ifelse(gender == "Male", 1, 0)) %>% 
  impute(what = "mean") %>% 
  as.data.frame()


train = sample(seq(length(x$EE)), 0.7 * length(x$EE))
test = seq(length(x$EE))[-train]

xTrain = x[train, ]
xTest = x[test, ]

reg <- lm(formula = EE ~ ., data = xTrain)

summary(reg)
```

```
## 
## Call:
## lm(formula = EE ~ ., data = xTrain)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -6.0804 -1.6198 -0.3266  1.0878 10.0935 
## 
## Coefficients:
##                                       Estimate Std. Error t value Pr(>|t|)
## (Intercept)                         -1.7471873  1.7202358  -1.016  0.30988
## age                                  0.0059368  0.0065958   0.900  0.36816
## gender                               0.5996433  0.1424354   4.210 2.64e-05
## height                               0.0009174  0.0091810   0.100  0.92041
## weight                               0.0255282  0.0050460   5.059 4.51e-07
## Applewatch.Steps                     0.0010492  0.0004889   2.146  0.03197
## Applewatch.Heart                     0.0113924  0.0073979   1.540  0.12369
## Applewatch.Calories                 -0.0494530  0.0176823  -2.797  0.00520
## Applewatch.Distance                  2.4912141  1.2654695   1.969  0.04911
## EntropyApplewatchHeartPerDay         0.0134865  0.0544326   0.248  0.80434
## EntropyApplewatchStepsPerDay        -0.1263450  0.0408537  -3.093  0.00201
## RestingApplewatchHeartrate           0.0017546  0.0032505   0.540  0.58937
## CorrelationApplewatchHeartrateSteps  0.1673602  0.6549561   0.256  0.79834
## NormalizedApplewatchHeartrate       -0.0417063  0.0150200  -2.777  0.00553
## ApplewatchIntensity                  9.9190128  1.9773737   5.016 5.63e-07
## SDNormalizedApplewatchHR            -0.0086564  0.0364764  -0.237  0.81243
## ApplewatchStepsXDistance            -0.0042229  0.0012970  -3.256  0.00114
##                                        
## (Intercept)                            
## age                                    
## gender                              ***
## height                                 
## weight                              ***
## Applewatch.Steps                    *  
## Applewatch.Heart                       
## Applewatch.Calories                 ** 
## Applewatch.Distance                 *  
## EntropyApplewatchHeartPerDay           
## EntropyApplewatchStepsPerDay        ** 
## RestingApplewatchHeartrate             
## CorrelationApplewatchHeartrateSteps    
## NormalizedApplewatchHeartrate       ** 
## ApplewatchIntensity                 ***
## SDNormalizedApplewatchHR               
## ApplewatchStepsXDistance            ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 2.204 on 2552 degrees of freedom
## Multiple R-squared:  0.2056,	Adjusted R-squared:  0.2007 
## F-statistic: 41.29 on 16 and 2552 DF,  p-value: < 2.2e-16
```

```r
ee_predict <- predict(reg, newdata = xTest)
```

##### Device name as a feature on Fitbit and AppleWatch interpolated data - Random Forest and SVM

```r
aggregated_data_appended <- fread(paste0(path, "aggregated_fitbit_applewatch_jaeger_appended.csv"), data.table = F)[ ,-1]
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


print(
    "--------------------------------------------Random Forest--------------------------------------------"
  )
```

```
## [1] "--------------------------------------------Random Forest--------------------------------------------"
```

```r
  rfModel <- randomForest(x = xTrain, y = yTrain, ntree = 500)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(rfModel[["predicted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            786             60             35             15
##   Running 3 METs    51            572             14              1
##   Running 5 METs    21             17            615             17
##   Running 7 METs    14              1             19            731
##   Self Pace walk    34             15              4              1
##   Sitting           46              5             29             23
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      25      57
##   Running 3 METs             13       5
##   Running 5 METs             11      51
##   Running 7 METs              2      22
##   Self Pace walk            551      28
##   Sitting                    31     462
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8479          
##                  95% CI : (0.8369, 0.8584)
##     No Information Rate : 0.2172          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.8164          
##                                           
##  Mcnemar's Test P-Value : 0.3082          
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8256                0.8537
## Specificity                0.9441                0.9774
## Pos Pred Value             0.8037                0.8720
## Neg Pred Value             0.9513                0.9737
## Prevalence                 0.2172                0.1528
## Detection Rate             0.1793                0.1305
## Detection Prevalence       0.2231                0.1496
## Balanced Accuracy          0.8848                0.9156
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.8589                0.9277
## Specificity                         0.9681                0.9839
## Pos Pred Value                      0.8402                0.9265
## Neg Pred Value                      0.9723                0.9841
## Prevalence                          0.1633                0.1797
## Detection Rate                      0.1403                0.1667
## Detection Prevalence                0.1670                0.1800
## Balanced Accuracy                   0.9135                0.9558
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8705         0.7392
## Specificity                         0.9781         0.9644
## Pos Pred Value                      0.8705         0.7752
## Neg Pred Value                      0.9781         0.9570
## Prevalence                          0.1444         0.1426
## Detection Rate                      0.1257         0.1054
## Detection Prevalence                0.1444         0.1359
## Balanced Accuracy                   0.9243         0.8518
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(rfModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            352             26              8              8
##   Running 3 METs    21            239              1              1
##   Running 5 METs    12              4            256              8
##   Running 7 METs     7              1             15            301
##   Self Pace walk    16              6              0              0
##   Sitting           19              4              6              8
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       8      20
##   Running 3 METs              4       2
##   Running 5 METs              7      20
##   Running 7 METs              0      15
##   Self Pace walk            223      10
##   Sitting                    14     238
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8559          
##                  95% CI : (0.8392, 0.8714)
##     No Information Rate : 0.2271          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.826           
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.8244                0.8536
## Specificity                0.9518                0.9819
## Pos Pred Value             0.8341                0.8918
## Neg Pred Value             0.9486                0.9746
## Prevalence                 0.2271                0.1489
## Detection Rate             0.1872                0.1271
## Detection Prevalence       0.2245                0.1426
## Balanced Accuracy          0.8881                0.9177
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                         0.8951                0.9233
## Specificity                         0.9680                0.9755
## Pos Pred Value                      0.8339                0.8879
## Neg Pred Value                      0.9809                0.9838
## Prevalence                          0.1521                0.1734
## Detection Rate                      0.1362                0.1601
## Detection Prevalence                0.1633                0.1803
## Balanced Accuracy                   0.9316                0.9494
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8711         0.7803
## Specificity                         0.9803         0.9676
## Pos Pred Value                      0.8745         0.8235
## Neg Pred Value                      0.9797         0.9579
## Prevalence                          0.1362         0.1622
## Detection Rate                      0.1186         0.1266
## Detection Prevalence                0.1356         0.1537
## Balanced Accuracy                   0.9257         0.8740
```

```r
  print(
    "------------------------------------------------SVM--------------------------------------------------"
  )
```

```
## [1] "------------------------------------------------SVM--------------------------------------------------"
```

```r
  svmModel <- svm(xTrain, yTrain)
  print(
    "________________________________________________Train________________________________________________"
  )
```

```
## [1] "________________________________________________Train________________________________________________"
```

```r
  print(confusionMatrix(svmModel[["fitted"]], yTrain))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            699            146            113            198
##   Running 3 METs   115            311             94             18
##   Running 5 METs    23            104            390             89
##   Running 7 METs    30             14             43            437
##   Self Pace walk    62             81             64             18
##   Sitting           23             14             12             28
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                     136     278
##   Running 3 METs            155      72
##   Running 5 METs             74      54
##   Running 7 METs             20      59
##   Self Pace walk            236      53
##   Sitting                    12     109
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4977          
##                  95% CI : (0.4828, 0.5126)
##     No Information Rate : 0.2172          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3876          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.7342               0.46418
## Specificity                0.7462               0.87776
## Pos Pred Value             0.4452               0.40654
## Neg Pred Value             0.9101               0.90080
## Prevalence                 0.2172               0.15283
## Detection Rate             0.1594               0.07094
## Detection Prevalence       0.3581               0.17450
## Balanced Accuracy          0.7402               0.67097
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                        0.54469               0.55457
## Specificity                        0.90622               0.95384
## Pos Pred Value                     0.53134               0.72471
## Neg Pred Value                     0.91068               0.90717
## Prevalence                         0.16332               0.17974
## Detection Rate                     0.08896               0.09968
## Detection Prevalence               0.16743               0.13755
## Balanced Accuracy                  0.72545               0.75420
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.37283        0.17440
## Specificity                        0.92589        0.97632
## Pos Pred Value                     0.45914        0.55051
## Neg Pred Value                     0.89742        0.87673
## Prevalence                         0.14439        0.14256
## Detection Rate                     0.05383        0.02486
## Detection Prevalence               0.11724        0.04516
## Balanced Accuracy                  0.64936        0.57536
```

```r
  print(
    "________________________________________________Test_________________________________________________"
  )
```

```
## [1] "________________________________________________Test_________________________________________________"
```

```r
  pred <- predict(svmModel, xTest)
  print(confusionMatrix(pred, yTest))
```

```
## Confusion Matrix and Statistics
## 
##                 Reference
## Prediction       Lying Running 3 METs Running 5 METs Running 7 METs
##   Lying            284             71             40             87
##   Running 3 METs    58            110             42              7
##   Running 5 METs    10             41            140             43
##   Running 7 METs    19              9             24            169
##   Self Pace walk    47             35             32              7
##   Sitting            9             14              8             13
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      73     160
##   Running 3 METs             69      30
##   Running 5 METs             37      25
##   Running 7 METs              4      28
##   Self Pace walk             68      24
##   Sitting                     5      38
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4303          
##                  95% CI : (0.4078, 0.4531)
##     No Information Rate : 0.2271          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3039          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs
## Sensitivity                0.6651               0.39286
## Specificity                0.7034               0.87125
## Pos Pred Value             0.3972               0.34810
## Neg Pred Value             0.8773               0.89130
## Prevalence                 0.2271               0.14894
## Detection Rate             0.1511               0.05851
## Detection Prevalence       0.3803               0.16809
## Balanced Accuracy          0.6842               0.63205
##                      Class: Running 5 METs Class: Running 7 METs
## Sensitivity                        0.48951               0.51840
## Specificity                        0.90213               0.94595
## Pos Pred Value                     0.47297               0.66798
## Neg Pred Value                     0.90783               0.90350
## Prevalence                         0.15213               0.17340
## Detection Rate                     0.07447               0.08989
## Detection Prevalence               0.15745               0.13457
## Balanced Accuracy                  0.69582               0.73218
##                      Class: Self Pace walk Class: Sitting
## Sensitivity                        0.26562        0.12459
## Specificity                        0.91071        0.96889
## Pos Pred Value                     0.31925        0.43678
## Neg Pred Value                     0.88722        0.85109
## Prevalence                         0.13617        0.16223
## Detection Rate                     0.03617        0.02021
## Detection Prevalence               0.11330        0.04628
## Balanced Accuracy                  0.58817        0.54674
```

