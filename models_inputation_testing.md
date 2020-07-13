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
##   Lying            147             54             34             49
##   Running 3 METs     4             30             23              2
##   Running 5 METs    16             13             88             26
##   Running 7 METs     5              3             11            100
##   Self Pace walk    53             53             30             13
##   Sitting            7              4             13              5
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      34      74
##   Running 3 METs              3       9
##   Running 5 METs              5      16
##   Running 7 METs              3      11
##   Self Pace walk             99      43
##   Sitting                     1      16
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4376          
##                  95% CI : (0.4079, 0.4675)
##     No Information Rate : 0.2115          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3182          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.6336               0.19108               0.44221
## Specificity                0.7168               0.95638               0.91537
## Pos Pred Value             0.3750               0.42254               0.53659
## Neg Pred Value             0.8794               0.87622               0.88103
## Prevalence                 0.2115               0.14312               0.18140
## Detection Rate             0.1340               0.02735               0.08022
## Detection Prevalence       0.3573               0.06472               0.14950
## Balanced Accuracy          0.6752               0.57373               0.67879
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.51282               0.68276        0.09467
## Specificity                        0.96341               0.79832        0.96767
## Pos Pred Value                     0.75188               0.34021        0.34783
## Neg Pred Value                     0.90145               0.94293        0.85442
## Prevalence                         0.17776               0.13218        0.15406
## Detection Rate                     0.09116               0.09025        0.01459
## Detection Prevalence               0.12124               0.26527        0.04193
## Balanced Accuracy                  0.73812               0.74054        0.53117
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
##   Lying            408             43             22             12
##   Running 3 METs    39            344              5              1
##   Running 5 METs    13              6            358             15
##   Running 7 METs    12              1             14            382
##   Self Pace walk    25              7              3              2
##   Sitting           49              5             16             14
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      20      41
##   Running 3 METs             15      10
##   Running 5 METs              6      29
##   Running 7 METs              0      26
##   Self Pace walk            316      21
##   Sitting                    25     254
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8058          
##                  95% CI : (0.7899, 0.8209)
##     No Information Rate : 0.2134          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.7661          
##                                           
##  Mcnemar's Test P-Value : 0.2057          
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7473                0.8473                0.8565
## Specificity                0.9314                0.9675                0.9678
## Pos Pred Value             0.7473                0.8309                0.8384
## Neg Pred Value             0.9314                0.9711                0.9719
## Prevalence                 0.2134                0.1587                0.1633
## Detection Rate             0.1594                0.1344                0.1399
## Detection Prevalence       0.2134                0.1618                0.1669
## Balanced Accuracy          0.8393                0.9074                0.9121
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8967                0.8272        0.66667
## Specificity                         0.9752                0.9734        0.94995
## Pos Pred Value                      0.8782                0.8449        0.69972
## Neg Pred Value                      0.9793                0.9698        0.94217
## Prevalence                          0.1665                0.1493        0.14889
## Detection Rate                      0.1493                0.1235        0.09926
## Detection Prevalence                0.1700                0.1462        0.14185
## Balanced Accuracy                   0.9359                0.9003        0.80831
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
##   Lying            196             11             11              5
##   Running 3 METs    14            146              4              0
##   Running 5 METs     5              3            154              6
##   Running 7 METs     3              0             11            173
##   Self Pace walk     5              5              2              0
##   Sitting           18              1              3              3
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       6       8
##   Running 3 METs              7       1
##   Running 5 METs              3      15
##   Running 7 METs              0      10
##   Self Pace walk            123       8
##   Sitting                     8     129
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8396          
##                  95% CI : (0.8165, 0.8608)
##     No Information Rate : 0.2197          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8065          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8133                0.8795                0.8324
## Specificity                0.9521                0.9721                0.9649
## Pos Pred Value             0.8270                0.8488                0.8280
## Neg Pred Value             0.9477                0.9784                0.9660
## Prevalence                 0.2197                0.1513                0.1686
## Detection Rate             0.1787                0.1331                0.1404
## Detection Prevalence       0.2160                0.1568                0.1696
## Balanced Accuracy          0.8827                0.9258                0.8987
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9251                0.8367         0.7544
## Specificity                         0.9736                0.9789         0.9644
## Pos Pred Value                      0.8782                0.8601         0.7963
## Neg Pred Value                      0.9844                0.9748         0.9551
## Prevalence                          0.1705                0.1340         0.1559
## Detection Rate                      0.1577                0.1121         0.1176
## Detection Prevalence                0.1796                0.1304         0.1477
## Balanced Accuracy                   0.9494                0.9078         0.8594
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
##   Lying            345             82             39             49
##   Running 3 METs    82            225             46              5
##   Running 5 METs    29             35            263             26
##   Running 7 METs    13             11             16            308
##   Self Pace walk    51             44             23              1
##   Sitting           26              9             31             37
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      86      94
##   Running 3 METs             68      22
##   Running 5 METs             21      54
##   Running 7 METs             13      22
##   Self Pace walk            178      40
##   Sitting                    16     149
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5737          
##                  95% CI : (0.5542, 0.5929)
##     No Information Rate : 0.2134          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4843          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.6319               0.55419                0.6292
## Specificity                0.8261               0.89642                0.9229
## Pos Pred Value             0.4964               0.50223                0.6145
## Neg Pred Value             0.8922               0.91426                0.9273
## Prevalence                 0.2134               0.15866                0.1633
## Detection Rate             0.1348               0.08792                0.1028
## Detection Prevalence       0.2716               0.17507                0.1673
## Balanced Accuracy          0.7290               0.72531                0.7761
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.7230               0.46597        0.39108
## Specificity                         0.9648               0.92696        0.94536
## Pos Pred Value                      0.8042               0.52819        0.55597
## Neg Pred Value                      0.9458               0.90819        0.89873
## Prevalence                          0.1665               0.14928        0.14889
## Detection Rate                      0.1204               0.06956        0.05823
## Detection Prevalence                0.1497               0.13169        0.10473
## Balanced Accuracy                   0.8439               0.69647        0.66822
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
##   Lying            141             28             20             33
##   Running 3 METs    52             82             28              2
##   Running 5 METs     8             17            102             12
##   Running 7 METs     5              4             16            125
##   Self Pace walk    26             18             11              0
##   Sitting            9             17              8             15
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      32      38
##   Running 3 METs             29      10
##   Running 5 METs              8      24
##   Running 7 METs              7      19
##   Self Pace walk             65      19
##   Sitting                     6      61
## 
## Overall Statistics
##                                         
##                Accuracy : 0.5251        
##                  95% CI : (0.495, 0.555)
##     No Information Rate : 0.2197        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.4253        
##                                         
##  Mcnemar's Test P-Value : 2.421e-11     
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.5851               0.49398               0.55135
## Specificity                0.8236               0.87003               0.92434
## Pos Pred Value             0.4829               0.40394               0.59649
## Neg Pred Value             0.8758               0.90604               0.91037
## Prevalence                 0.2197               0.15132               0.16864
## Detection Rate             0.1285               0.07475               0.09298
## Detection Prevalence       0.2662               0.18505               0.15588
## Balanced Accuracy          0.7043               0.68200               0.73785
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.6684               0.44218        0.35673
## Specificity                         0.9440               0.92211        0.94060
## Pos Pred Value                      0.7102               0.46763        0.52586
## Neg Pred Value                      0.9327               0.91441        0.88787
## Prevalence                          0.1705               0.13400        0.15588
## Detection Rate                      0.1139               0.05925        0.05561
## Detection Prevalence                0.1604               0.12671        0.10574
## Balanced Accuracy                   0.8062               0.68214        0.64866
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
##   Lying            193             20              8              5
##   Running 3 METs    10            117              8              4
##   Running 5 METs     3             40            128             15
##   Running 7 METs     5              8              7            153
##   Self Pace walk    10             12              5              3
##   Sitting           11              0              9              8
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      18      52
##   Running 3 METs             28       1
##   Running 5 METs             18      11
##   Running 7 METs              8       3
##   Self Pace walk             79      13
##   Sitting                     6      83
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6833          
##                  95% CI : (0.6549, 0.7107)
##     No Information Rate : 0.2105          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6171          
##                                           
##  Mcnemar's Test P-Value : 2.734e-11       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8319                0.5939                0.7758
## Specificity                0.8816                0.9436                0.9072
## Pos Pred Value             0.6520                0.6964                0.5953
## Neg Pred Value             0.9516                0.9143                0.9583
## Prevalence                 0.2105                0.1788                0.1497
## Detection Rate             0.1751                0.1062                0.1162
## Detection Prevalence       0.2686                0.1525                0.1951
## Balanced Accuracy          0.8568                0.7688                0.8415
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.8138               0.50318        0.50920
## Specificity                         0.9661               0.95450        0.96379
## Pos Pred Value                      0.8315               0.64754        0.70940
## Neg Pred Value                      0.9619               0.92041        0.91878
## Prevalence                          0.1706               0.14247        0.14791
## Detection Rate                      0.1388               0.07169        0.07532
## Detection Prevalence                0.1670               0.11071        0.10617
## Balanced Accuracy                   0.8900               0.72884        0.73650
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
##   Lying            372             18              6              3
##   Running 3 METs    17            231              4              1
##   Running 5 METs     5              3            247              4
##   Running 7 METs    10              1              4            338
##   Self Pace walk    12              2              0              0
##   Sitting           11              0             10              5
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       8      17
##   Running 3 METs              5       0
##   Running 5 METs              2      12
##   Running 7 METs              1       8
##   Self Pace walk            236       8
##   Sitting                    11     213
## 
## Overall Statistics
##                                           
##                Accuracy : 0.897           
##                  95% CI : (0.8821, 0.9106)
##     No Information Rate : 0.234           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8753          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8712                0.9059                0.9114
## Specificity                0.9628                0.9828                0.9833
## Pos Pred Value             0.8774                0.8953                0.9048
## Neg Pred Value             0.9607                0.9847                0.9845
## Prevalence                 0.2340                0.1397                0.1485
## Detection Rate             0.2038                0.1266                0.1353
## Detection Prevalence       0.2323                0.1414                0.1496
## Balanced Accuracy          0.9170                0.9443                0.9474
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9630                0.8973         0.8256
## Specificity                         0.9837                0.9859         0.9764
## Pos Pred Value                      0.9337                0.9147         0.8520
## Neg Pred Value                      0.9911                0.9828         0.9714
## Prevalence                          0.1923                0.1441         0.1414
## Detection Rate                      0.1852                0.1293         0.1167
## Detection Prevalence                0.1984                0.1414         0.1370
## Balanced Accuracy                   0.9733                0.9416         0.9010
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
##   Lying            139              1              6              0
##   Running 3 METs    10            120              1              1
##   Running 5 METs     4              0            118              0
##   Running 7 METs     3              0              0            141
##   Self Pace walk     7              2              0              2
##   Sitting            2              0              3              6
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                       3       5
##   Running 3 METs              0       0
##   Running 5 METs              0       9
##   Running 7 METs              1       5
##   Self Pace walk             92       2
##   Sitting                     1      99
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9055          
##                  95% CI : (0.8828, 0.9251)
##     No Information Rate : 0.2107          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8861          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8424                0.9756                0.9219
## Specificity                0.9757                0.9818                0.9802
## Pos Pred Value             0.9026                0.9091                0.9008
## Neg Pred Value             0.9587                0.9954                0.9847
## Prevalence                 0.2107                0.1571                0.1635
## Detection Rate             0.1775                0.1533                0.1507
## Detection Prevalence       0.1967                0.1686                0.1673
## Balanced Accuracy          0.9091                0.9787                0.9510
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9400                0.9485         0.8250
## Specificity                         0.9858                0.9810         0.9819
## Pos Pred Value                      0.9400                0.8762         0.8919
## Neg Pred Value                      0.9858                0.9926         0.9688
## Prevalence                          0.1916                0.1239         0.1533
## Detection Rate                      0.1801                0.1175         0.1264
## Detection Prevalence                0.1916                0.1341         0.1418
## Balanced Accuracy                   0.9629                0.9648         0.9035
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
##   Lying            344             34             28             55
##   Running 3 METs    25            133             19             11
##   Running 5 METs     8             68            210             42
##   Running 7 METs    33              2              8            239
##   Self Pace walk    12             18              5              3
##   Sitting            5              0              1              1
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      39     155
##   Running 3 METs             86      30
##   Running 5 METs             39      17
##   Running 7 METs              1      22
##   Self Pace walk             96       9
##   Sitting                     2      25
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5737          
##                  95% CI : (0.5506, 0.5965)
##     No Information Rate : 0.234           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4772          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8056               0.52157                0.7749
## Specificity                0.7775               0.89108                0.8880
## Pos Pred Value             0.5252               0.43750                0.5469
## Neg Pred Value             0.9291               0.91979                0.9577
## Prevalence                 0.2340               0.13973                0.1485
## Detection Rate             0.1885               0.07288                0.1151
## Detection Prevalence       0.3589               0.16658                0.2104
## Balanced Accuracy          0.7916               0.70633                0.8315
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.6809               0.36502        0.09690
## Specificity                         0.9552               0.96991        0.99426
## Pos Pred Value                      0.7836               0.67133        0.73529
## Neg Pred Value                      0.9263               0.90071        0.86991
## Prevalence                          0.1923               0.14411        0.14137
## Detection Rate                      0.1310               0.05260        0.01370
## Detection Prevalence                0.1671               0.07836        0.01863
## Balanced Accuracy                   0.8181               0.66746        0.54558
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
##   Lying            123             19             19             35
##   Running 3 METs    13             39             14              3
##   Running 5 METs     3             44             87             30
##   Running 7 METs    18              2              5             81
##   Self Pace walk     7             18              3              0
##   Sitting            1              1              0              1
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      11      69
##   Running 3 METs             39       8
##   Running 5 METs             16      12
##   Running 7 METs              3      24
##   Self Pace walk             27       3
##   Sitting                     1       4
## 
## Overall Statistics
##                                           
##                Accuracy : 0.461           
##                  95% CI : (0.4257, 0.4967)
##     No Information Rate : 0.2107          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.342           
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7455               0.31707                0.6797
## Specificity                0.7524               0.88333                0.8397
## Pos Pred Value             0.4457               0.33621                0.4531
## Neg Pred Value             0.9172               0.87406                0.9306
## Prevalence                 0.2107               0.15709                0.1635
## Detection Rate             0.1571               0.04981                0.1111
## Detection Prevalence       0.3525               0.14815                0.2452
## Balanced Accuracy          0.7489               0.60020                0.7597
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.5400               0.27835       0.033333
## Specificity                         0.9179               0.95481       0.993967
## Pos Pred Value                      0.6090               0.46552       0.500000
## Neg Pred Value                      0.8938               0.90345       0.850323
## Prevalence                          0.1916               0.12388       0.153257
## Detection Rate                      0.1034               0.03448       0.005109
## Detection Prevalence                0.1699               0.07407       0.010217
## Balanced Accuracy                   0.7289               0.61658       0.513650
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
##   Lying            794             63             38             20
##   Running 3 METs    53            545             12              1
##   Running 5 METs    20              9            608             15
##   Running 7 METs    17              2             20            723
##   Self Pace walk    33             14              5              1
##   Sitting           60              5             20             23
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      28      54
##   Running 3 METs             12       4
##   Running 5 METs              9      41
##   Running 7 METs              0      28
##   Self Pace walk            545      24
##   Sitting                    42     496
## 
## Overall Statistics
##                                          
##                Accuracy : 0.8465         
##                  95% CI : (0.8355, 0.857)
##     No Information Rate : 0.2229         
##     P-Value [Acc > NIR] : < 2e-16        
##                                          
##                   Kappa : 0.8147         
##                                          
##  Mcnemar's Test P-Value : 0.06629        
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8127                0.8542                0.8649
## Specificity                0.9404                0.9781                0.9745
## Pos Pred Value             0.7964                0.8692                0.8661
## Neg Pred Value             0.9460                0.9752                0.9742
## Prevalence                 0.2229                0.1455                0.1604
## Detection Rate             0.1811                0.1243                0.1387
## Detection Prevalence       0.2274                0.1430                0.1601
## Balanced Accuracy          0.8766                0.9162                0.9197
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9234                0.8569         0.7666
## Specificity                         0.9814                0.9795         0.9599
## Pos Pred Value                      0.9152                0.8762         0.7678
## Neg Pred Value                      0.9833                0.9758         0.9596
## Prevalence                          0.1786                0.1451         0.1476
## Detection Rate                      0.1649                0.1243         0.1131
## Detection Prevalence                0.1802                0.1419         0.1474
## Balanced Accuracy                   0.9524                0.9182         0.8632
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
##   Lying            330             24             12              7
##   Running 3 METs    20            273              7              1
##   Running 5 METs     9              4            258              3
##   Running 7 METs     8              1             10            316
##   Self Pace walk    17              8              1              0
##   Sitting           18              2             11              4
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      10      19
##   Running 3 METs              4       3
##   Running 5 METs              4      24
##   Running 7 METs              0      12
##   Self Pace walk            225       8
##   Sitting                    10     217
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8612          
##                  95% CI : (0.8447, 0.8765)
##     No Information Rate : 0.2138          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8326          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8209                0.8750                0.8629
## Specificity                0.9513                0.9777                0.9722
## Pos Pred Value             0.8209                0.8864                0.8543
## Neg Pred Value             0.9513                0.9752                0.9740
## Prevalence                 0.2138                0.1660                0.1590
## Detection Rate             0.1755                0.1452                0.1372
## Detection Prevalence       0.2138                0.1638                0.1606
## Balanced Accuracy          0.8861                0.9263                0.9175
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.9547                0.8893         0.7668
## Specificity                         0.9800                0.9791         0.9718
## Pos Pred Value                      0.9107                0.8687         0.8282
## Neg Pred Value                      0.9902                0.9827         0.9592
## Prevalence                          0.1761                0.1346         0.1505
## Detection Rate                      0.1681                0.1197         0.1154
## Detection Prevalence                0.1846                0.1378         0.1394
## Balanced Accuracy                   0.9673                0.9342         0.8693
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
##   Lying            729            164            133            190
##   Running 3 METs    52            237             63             15
##   Running 5 METs    29             85            386             86
##   Running 7 METs    31             12             41            437
##   Self Pace walk   103            123             68             23
##   Sitting           33             17             12             32
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                     163     290
##   Running 3 METs            113      27
##   Running 5 METs             54      55
##   Running 7 METs             17      57
##   Self Pace walk            280      96
##   Sitting                     9     122
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4998          
##                  95% CI : (0.4849, 0.5147)
##     No Information Rate : 0.2229          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3886          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.7462               0.37147               0.54908
## Specificity                0.7241               0.92792               0.91606
## Pos Pred Value             0.4368               0.46746               0.55540
## Neg Pred Value             0.9087               0.89657               0.91407
## Prevalence                 0.2229               0.14553               0.16036
## Detection Rate             0.1663               0.05406               0.08805
## Detection Prevalence       0.3807               0.11565               0.15853
## Balanced Accuracy          0.7351               0.64970               0.73257
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.55811               0.44025        0.18856
## Specificity                        0.95612               0.88981        0.97244
## Pos Pred Value                     0.73445               0.40404        0.54222
## Neg Pred Value                     0.90868               0.90355        0.87377
## Prevalence                         0.17860               0.14507        0.14758
## Detection Rate                     0.09968               0.06387        0.02783
## Detection Prevalence               0.13572               0.15807        0.05132
## Balanced Accuracy                  0.75712               0.66503        0.58050
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
##   Lying            281             82             54             79
##   Running 3 METs    29            111             41              7
##   Running 5 METs    15             37            138             35
##   Running 7 METs    15              3             24            183
##   Self Pace walk    48             71             34              9
##   Sitting           14              8              8             18
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      69     140
##   Running 3 METs             56      18
##   Running 5 METs             23      28
##   Running 7 METs             10      26
##   Self Pace walk             93      35
##   Sitting                     2      36
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4479          
##                  95% CI : (0.4252, 0.4707)
##     No Information Rate : 0.2138          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3277          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.6990               0.35577                0.4615
## Specificity                0.7131               0.90370                0.9127
## Pos Pred Value             0.3986               0.42366                0.5000
## Neg Pred Value             0.8970               0.87577                0.8996
## Prevalence                 0.2138               0.16596                0.1590
## Detection Rate             0.1495               0.05904                0.0734
## Detection Prevalence       0.3750               0.13936                0.1468
## Balanced Accuracy          0.7061               0.62973                0.6871
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.55287               0.36759        0.12721
## Specificity                        0.94964               0.87892        0.96869
## Pos Pred Value                     0.70115               0.32069        0.41860
## Neg Pred Value                     0.90859               0.89937        0.86232
## Prevalence                         0.17606               0.13457        0.15053
## Detection Rate                     0.09734               0.04947        0.01915
## Detection Prevalence               0.13883               0.15426        0.04574
## Balanced Accuracy                  0.75126               0.62325        0.54795
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
##   Lying            222            161            139            117
##   Running 3 METs     0              0              0              0
##   Running 5 METs     9              6             26             16
##   Running 7 METs     8              7             10             60
##   Self Pace walk     0              0              0              0
##   Sitting            0              0              0              0
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                     141     143
##   Running 3 METs              0       0
##   Running 5 METs              5      14
##   Running 7 METs              9       9
##   Self Pace walk              0       0
##   Sitting                     0       0
## 
## Overall Statistics
##                                          
##                Accuracy : 0.2795         
##                  95% CI : (0.2532, 0.307)
##     No Information Rate : 0.2169         
##     P-Value [Acc > NIR] : 5.82e-07       
##                                          
##                   Kappa : 0.0892         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.9289                0.0000               0.14857
## Specificity                0.1877                1.0000               0.94606
## Pos Pred Value             0.2405                   NaN               0.34211
## Neg Pred Value             0.9050                0.8421               0.85478
## Prevalence                 0.2169                0.1579               0.15880
## Detection Rate             0.2015                0.0000               0.02359
## Detection Prevalence       0.8376                0.0000               0.06897
## Balanced Accuracy          0.5583                0.5000               0.54732
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                        0.31088                0.0000         0.0000
## Specificity                        0.95270                1.0000         1.0000
## Pos Pred Value                     0.58252                   NaN            NaN
## Neg Pred Value                     0.86687                0.8593         0.8494
## Prevalence                         0.17514                0.1407         0.1506
## Detection Rate                     0.05445                0.0000         0.0000
## Detection Prevalence               0.09347                0.0000         0.0000
## Balanced Accuracy                  0.63179                0.5000         0.5000
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
##   Lying            192             11             10             13
##   Running 3 METs     2             59             17              4
##   Running 5 METs     5             51            141             27
##   Running 7 METs     1              5             10            137
##   Self Pace walk     4             28              6              6
##   Sitting           26             12             13              8
##                 Reference
## Prediction       Self Pace walk Sitting
##   Lying                      19     112
##   Running 3 METs             31       3
##   Running 5 METs             51      11
##   Running 7 METs              7       0
##   Self Pace walk             31       5
##   Sitting                    16      28
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5336          
##                  95% CI : (0.5036, 0.5634)
##     No Information Rate : 0.2087          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4317          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: Lying Class: Running 3 METs Class: Running 5 METs
## Sensitivity                0.8348               0.35542                0.7157
## Specificity                0.8108               0.93910                0.8398
## Pos Pred Value             0.5378               0.50862                0.4930
## Neg Pred Value             0.9490               0.89148                0.9314
## Prevalence                 0.2087               0.15064                0.1788
## Detection Rate             0.1742               0.05354                0.1279
## Detection Prevalence       0.3240               0.10526                0.2595
## Balanced Accuracy          0.8228               0.64726                0.7778
##                      Class: Running 7 METs Class: Self Pace walk Class: Sitting
## Sensitivity                         0.7026               0.20000        0.17610
## Specificity                         0.9746               0.94826        0.92047
## Pos Pred Value                      0.8562               0.38750        0.27184
## Neg Pred Value                      0.9384               0.87867        0.86887
## Prevalence                          0.1770               0.14065        0.14428
## Detection Rate                      0.1243               0.02813        0.02541
## Detection Prevalence                0.1452               0.07260        0.09347
## Balanced Accuracy                   0.8386               0.57413        0.54828
```


