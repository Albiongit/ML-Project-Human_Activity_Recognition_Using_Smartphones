# ML-Project-Human_Activity_Recognition_Using_Smartphones

## Dataset information
The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data. 

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

## Phase 1
To execute this project, first you need to install the libraries pandas and numpy by writing the following command in the terminal:
```bash
pip install pandas
pip install numpy
```

and then write the following command in the terminal:
```bash
python script.py
```

In this project, we have used a dataset borrowed from the UC Irvine Machine Learning Repository at the following link: [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). It contains 561 columns which are included in the Rar file uploaded on the corresponding page found in the "features.txt" file.

We will develop a prediction about human unexpected behaviors: whether a person will engage in unexpected behaviors during the day or predict patterns of human activities based on data recorded by a smartphone.

The classification algorithms that we intend to use to find the one that best fits our needs and yields better results are:
1.  Naive Bayes
2.  Decision Tree
3.  Logistic Regression
with the possibility to add any more suitable algorithm in the future during work.
