# fitbit_apple_watch_classification

## Submitted
Journal for the Measurement of Physical Behaviour

## Title 
Using machine learning methods to predict physical activity types with Apple Watch and Fitbit data using indirect calorimetry as the criterion

## Authors
Fuller D, Rahimipour Anaraki J, Simango B, Dorani F, Bozorgi A, Luan H, Basset F. 

## Abstract
### Objectives
There is considerable promise for using commercial wearable devices for measuring physical activity at the population level. The objective of this study was to examine whether commercial wearable devices could accurately predict lying, sitting, and intensity level of other activities in a lab-based protocol. 

### Methods
We recruited a convenience sample of 49 participants (23 men and 26 women) to wear three devices, an Apple Watch Series 2, a Fitbit Charge HR2, and and iPhone 6S. Participants completed a 65-minute protocol consisting of 40 minutes of total treadmill time and 25 minutes of sitting or lying time. Indirect calorimetry was used to measure energy expenditure. The outcome variable for the study was the activity class; lying, sitting, walking self-paced, and running 3 METs, 5 METs, and 7 METs. Minute-by-minute heart rate, steps, distance, and calories from Apple Watch and Fitbit were included in four different machine learning models. 

### Results
Our dataset included 3656 and 2608 minutes of Apple Watch and Fitbit data, respectively. We tested decision trees, support vector machines, random forest, and rotation forest models. Random Forest and Rotation Forest models had similar model accuracies. Rotation Forest accuracies were 82.6% for Apple Watch and 89.3% for Fitbit. Classification accuracies for Apple Watch data ranged from 72.5% for sitting to 89.0% for 7 METs. For Fitbit, accuracies varied between 86.2% for sitting to 92.6% for 7 METs. 

### Conclusion
This study demonstrated that commercial wearable devices, Apple Watch and Fitbit, were able to predict physical activity types with a reasonable accuracy. The results support the use of minute-by-minute data from Apple Watch and Fitbit combined with machine learning approaches for scalable physical activity type classification at the population level. 

## Code 

1. For converting indirect calorimetry data from breath-by-breath data to second-by-second: [https://github.com/walkabillylab/jaeger_analysis](https://github.com/walkabillylab/jaeger_analysis)

2. For modeling decision tree, random forest, and support vector machine analyses: [https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/model_results.Rmd](https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/model_results.Rmd)

## Results

1. Random Forest, SVM, and Decision Tree Models
[https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/model_results.md](https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/model_results.md)

2. Rotation Forest (Apple Watch) [https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/AppleWatch_RotationForestOutput.txt](https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/AppleWatch_RotationForestOutput.txt)

3. Rotation Forest (Fitbit) [https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/Fitbit_RotationForestOutput.txt](https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/Fitbit_RotationForestOutput.txt)

## Models

1. Rotation Forest in Weka for Fitbit data: [https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/Fitbit_RotationForestModel.model](https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/Fitbit_RotationForestModel.model)

2. Rotation Forest in Weka for Apple Watch data: [https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/AppleWatch_RotationForestModel.model](https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/AppleWatch_RotationForestModel.model)

3. Rotation Forest Model with combined Fitbit and Apple Watch data: [https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/AppleWatch_Fitbit_RotationForestModel.model](https://github.com/walkabillylab/fitbit_apple_watch_classification/blob/master/AppleWatch_Fitbit_RotationForestModel.model)

## Data
To be published soon. 
