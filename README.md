# Project in the subject "Machine Learning" - Computer Engineering Master's Program 23/24 - [University of Prishtina](https://fiek.uni-pr.edu)

## Dataset information
The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data. 

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

## Faza 1
Per te ekzekutuar kete paze te projektit fillimisht duhet te instaloni librarite pandas dhe numpy duke shkruar ne terminal:
```bash
pip install pandas
pip install numpy
```

dhe me pas te shkruani ne terminal komanden:
```bash
python script.py
```

Ne kete projekt ne kemi perdorur nje dataset te huazuar nga UC Irvine Machine Learning Repository ne linkun ne vijim: [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). I cili i permban 561 kolona e cila ne Rar fajllin e ngarkuar faqja perkatese gjenden tek fajlli "features.txt".


