# Project in the subject "Machine Learning" - Computer Engineering Master's Program 23/24 - [University of Prishtina](https://fiek.uni-pr.edu)

## Overview
  This is a project done by Albion Ademi & Lirim Maloku in the subject "Machine Learning", supervised by professor [Lule Ahmedi](https://staff.uni-pr.edu/profile/luleahmedi) & teaching assistant [Mërgim Hoti](https://staff.uni-pr.edu/profile/m%C3%ABrgimhoti). The requirements will be met in specific phases, with each phase being evaluated and improved in the next phases based on the given feedback.

## Dataset information
The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data. 

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

## Phase 1
The script is written in Python so you will need to install Python in your machine. More information about it you can find [Python's Official Page](https://www.python.org).
After installing Python, these are the libraries required to run the scripts:
- pandas: Used for data manipulation and analysis.
- numpy: A library for numerical operations in Python.
- sklearn.ensemble.RandomForestClassifier: Used to solve for regression or classification problems
- sklearn.ensemble.GradientBoostingClassifier: A library that repeatedly selects a function that leads in the direction of a weak hypothesis or negative gradient so that it can minimize a loss function.
- sklearn.metrics.precision_score: This library implements several loss, score, and utility functions to measure classification performance.
- sklearn.metrics.recall_score: This library implements several loss, score, and utility functions to measure classification performance.
- sklearn.metrics.f1_score: Used to compute the F1 score, also known as balanced F-score or F-measure.
- matplotlib.pyplot: A library for creating static, animated, and interactive visualizations.

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
```

and then write the following command in the terminal:
```bash
python script.py
```

In this project, we have used a dataset borrowed from the UC Irvine Machine Learning Repository at the following link: [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). It contains 561 columns which are included in the Rar file uploaded on the corresponding page found in the "features.txt" file.

Data preprocessing:
1.  Loading the data:
-  The dataset is read from a path and loaded into a pandas DataFrame using the pd.read_csv function.
2.  Checking for Null Values
- A null value count check is performed on each column of the dataset.
3.  Handling Null Values:
- The number of null values was zero.

We will develop a prediction about human unexpected behaviors: whether a person will engage in unexpected behaviors during the day or predict patterns of human activities based on data recorded by a smartphone.

The classification algorithms that we intend to use to find the one that best fits our needs and yields better results are:
1.  Random Forest Algorithm
2.  Gradient Boosting Machines Algorithm

with the possibility to add any more suitable algorithm in the future during work.

## Phase 2

In this phase we focused on training the model & doing the evaluation of it afterwards. We will train the model with two algorithms, best suitable for what we want to achieve.
On the other hand the evaluation was done by calculating precision, recall and F1-Score, but also by visualizing the prediction accuracy by comparing it against a ground truth which in our case is the *ACTIVITY* label in our csv containing the test data.

### • Feature Selection •

The biggest challenge in our project was feature selection, given the vast array of potential features available. To address this challenge, we researched about gyroscope functionality and got insights from research, particularly from the paper [A Public Domain Dataset for Human Activity Recognition using Smartphones](https://www.semanticscholar.org/paper/A-Public-Domain-Dataset-for-Human-Activity-using-Anguita-Ghio/83de43bc849ad3d9579ccf540e6fe566ef90a58e). This paper provided valuable guidance on relevant features for activity recognition tasks using smartphone sensor data. With 561 features in total, exploring all combinations was impractical. Instead, we focused on a targeted selection based on our understanding and the research findings. This approach allowed us to prioritize the most informative features for accurate activity recognition. 

The selected features were chosen based on their relevance to human activity recognition and their ability to capture important aspects of motion and orientation. The following features were chosen:

- **tBodyAcc-mean()-X, tBodyAcc-mean()-Y, tBodyAcc-mean()-Z**: Mean value of body acceleration signals in the X, Y, and Z directions.
- **fBodyGyro-skewness()-Y, fBodyGyro-skewness()-Z, fBodyGyro-skewness()-X**: Skewness of body gyroscope signals in the X, Y, and Z directions.
- **tBodyGyro-mean()-X, tBodyGyro-mean()-Y, tBodyGyro-mean()-Z**: Mean value of body gyroscope signals in the X, Y, and Z directions.
- **tBodyAccJerk-mean()-X, tBodyAccJerk-mean()-Y, tBodyAccJerk-mean()-Z**: Mean value of body acceleration jerk signals in the X, Y, and Z directions.
- **angle(X,gravityMean), angle(Y,gravityMean), angle(Z,gravityMean)**: Angle between each axis and the gravity vector.
- **tBodyAcc-sma()**: Signal magnitude area of body acceleration.

### • Test Data Preparation • 

Important to mention is that test data makes up for roughly 30% of the dataset, whereas the other 70% of the dataset was used only for training.
Test data was created by combining CSV files similar to the process used for training data. The test data includes features extracted from smartphone sensor readings along with the corresponding activity labels. The test data is exported in a file called **test-data.csv**.

### • Chosen Algorithms - Supervised Learning •

#### Random Forest Algorithm
The Random Forest algorithm was chosen for its ability to handle high-dimensional data and its robustness against overfitting. By using an ensemble of decision trees, Random Forest can effectively capture complex relationships between features and target variables. During training, multiple decision trees are built using random subsets of the data and random subsets of the features, which helps to reduce variance and improve generalization. In our evaluation, the Random Forest model demonstrated strong performance, achieving high precision, recall, and F1-score on the test data.

#### Gradient Boosting Machines (GBM) Algorithm
Gradient Boosting Machines (GBM) was selected for its ability to build strong predictive models by iteratively improving the weaknesses of previous models. GBM sequentially builds multiple weak learners, such as decision trees, and combines them to create a single strong learner. By minimizing a loss function, GBM focuses on areas where previous models performed poorly, leading to improved accuracy over time. In our evaluation, the GBM model also exhibited high precision, recall, and F1-score on the test data.


### • Model Evaluation By Calculating Precision, Recall & F1 Score •

This is the first method we used for our model evaluation and the results are as below.

#### Random Forest Model Metrics:
```bash
Accuracy: 81.74414658975229
Precision: 0.8197205741687814
Recall: 0.8174414658975229
F1-Score: 0.8172982748951982
```

#### Gradient Boosting Machines Model Metrics:
```bash
Accuracy: 79.84390906006108
Precision: 0.8000325277318522
Recall: 0.7984390906006108
F1-Score: 0.7988625485684024
```

### • Model Evaluation By Comparison • 
Another method we used to evaluate our trained model is by comparing it to the correct activity for the rows that we have in our **ACTIVITY** column.
We have visualized this information to make it easier to understand.

In the image below you can find the overall accuracy of the algorithms in predicting subject activity.
![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/34185066/d79d48d8-9101-460d-8027-c76f281d7dfd)

In the image below we can see the percentage of what activity the algorithms got wrong. This is expressed as a percentage of the activities the algorithms predicted wrongly.
![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/34185066/7463ea38-63d9-42c9-b549-3c4983a4ca31)

As a last step we took to evaluate our model and check for algorithm's accuracy is to visualize the pairs of activities, so **CORRECT ACTIVITY - WRONGLY PREDICTED ACTIVITY**, for each algorithm separately.

**Random Forest**
![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/34185066/b16169bb-bee0-469d-a6c3-c98a7b1d19fc)

**GBM**
![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/34185066/76cda78e-514d-4710-8e3b-3b1787e8d6b0)

### • Conclusions after the evaluation •
The most common instances where the algorithms misclassified activities were when the activities involved variations of walking, such as walking, walking upstairs, and walking downstairs.
Similarly, misclassifications occurred between standing and sitting activities. These errors can be attributed to the inherent similarity between these activities, particularly from the perspective of smartphone gyroscope data.

When a subject is walking, whether it's on a flat surface, walking upstairs, or walking downstairs, the movements captured by the gyroscope sensor may exhibit similar patterns. The variations in the orientation and intensity of movements can be subtle, making it challenging for the algorithms to differentiate between these activities solely based on gyroscope data. 
Similarly, distinguishing between standing and sitting activities can be difficult because both activities involve minimal movement and can result in similar angles and gravity-related measurements.

In essence, the gyroscope data may not provide sufficient discriminatory information to accurately classify these activities, especially when the differences between them are nuanced. Factors such as the angle of inclination, acceleration, and gravitational forces may overlap or be indistinguishable in certain scenarios, leading to misclassifications by the algorithms. 
As a result, additional contextual information or sensor data from complementary sources may be necessary to improve the accuracy of activity recognition in such cases.

Despite the challenges posed by the subtle distinctions in the gyroscope data, the models achieved an impressive ~80% accuracy in predicting activity. 
Achieving an 80% accuracy demonstrates the effectiveness of the selected features and the predictive capabilities of the Random Forest and Gradient Boosting Machines algorithms in discerning patterns and trends within the data.
While some misclassifications occurred, particularly in activities with similar movement patterns, the overall performance of the models indicates a high level of predictive power. 
This level of accuracy is considered very good, especially in the context of complex human activity recognition tasks using sensor data.

## Phase 3

In this phase we've added another algorithm to potentially reach higher levels of accuracy in predicting the activities of the subjects. We added SVM (Support Vector Machine) and got the results like below.

#### Support Vector Machine Model Metrics:
```bash
Accuracy: 84.5945028842891
Precision: 0.8544101419075175
Recall: 0.845945028842891
F1-Score: 0.8463846813290467
```

Accuracy chart of algorithms, SVM included.
![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/62037447/2ea50ff0-11a6-4267-bd20-570f32d03488)

Percentage of missed predictions activity pairs by SVM. 
![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/62037447/075908d5-0095-4b73-8a2a-fb414b7583a9)

### • GUI application to use the trained model •

We've developed a simple GUI app where you can upload a csv of data and the model will then predict the activity and visualize useful information. This app uses the SVM trained model to make the predictions because that's the one with the best evaluation scores.

![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/62037447/cd738966-9400-40af-b577-5e8a0dc0f85a)

Upon successful upload the subjects will appear below as icons. 

![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/62037447/6476b6c9-c96a-4119-b455-155a2eb246cf)

If for example we click on Subject 2 a window appears with the distribution of activities for that specific subject.

![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/62037447/0e158e4a-89e4-4f96-9faa-ce52c6fdf96f)

If we choose "More statistics" option a new window appears where we see by default the distribution of activities by all subjects.

![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/62037447/cee08e6d-c2a6-4b84-a712-4b56efc8d426)

There's also the option to see data of subjects based on a specific activity like "Standing" for example, like in the image below.

![image](https://github.com/Albiongit/ML-Project-Human_Activity_Recognition_Using_Smartphones/assets/62037447/ae76bb6f-f4fb-4fa2-b475-a02dc27370f7)


## Current Status

The project has completed all phases of requirements, and we've successfully trained a model to make predictions of activities based on gyroscope data. In addition we've developed a simple GUI app to make use of the trained model and make a few useful visualizations based on the predicted data.

## Contributors

Contributors: Albion Ademi & Lirim Maloku.
If you'd like to contribute or improve the project, feel free to raise a pull request. 
