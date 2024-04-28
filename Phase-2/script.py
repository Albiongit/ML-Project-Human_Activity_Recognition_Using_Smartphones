import csv
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

base_dir = 'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset'
data_dir = 'train'
test_dir = 'test'

activity_labels_file = os.path.join(base_dir, 'activity_labels.txt')
features_file = os.path.join(base_dir, 'features.txt')

train_subject_file = os.path.join(base_dir, data_dir, 'subject_train.txt')
train_activity_file = os.path.join(base_dir, data_dir, 'y_train.txt')
train_data_file = os.path.join(base_dir, data_dir, 'X_train.txt')

test_subject_file = os.path.join(base_dir, test_dir, 'subject_test.txt')
test_activity_file = os.path.join(base_dir, test_dir, 'y_test.txt')
test_data_file = os.path.join(base_dir, test_dir, 'X_test.txt')

# Selected features
selected_features = [
    "tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z",
    "fBodyGyro-skewness()-Y", "fBodyGyro-skewness()-Z", "fBodyGyro-skewness()-X",
    "tBodyGyro-mean()-X", "tBodyGyro-mean()-Y", "tBodyGyro-mean()-Z",
    "tBodyAccJerk-mean()-X", "tBodyAccJerk-mean()-Y", "tBodyAccJerk-mean()-Z",
    "angle(X,gravityMean)", "angle(Y,gravityMean)", "angle(Z,gravityMean)", "tBodyAcc-sma()",
]

#"tGravityAcc-mean()-Y", - affects sitting/standing detection

# Function to read data from a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Function to write data to a CSV file
def write_csv(output_file, rows, headers):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

# Function to parse activity labels
def read_activity_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = {int(line.split()[0]): line.split()[1] for line in lines}
    return labels

# Function to read feature names
def read_feature_names(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        feature_names = [line.split()[1] for line in lines]
    return feature_names

# Get data from input files
train_subjects = read_file(train_subject_file)
train_activities = read_file(train_activity_file)
train_data = read_file(train_data_file)

# Get data from test input files
test_subjects = read_file(test_subject_file)
test_activities = read_file(test_activity_file)
test_data = read_file(test_data_file)

# Parse activity labels
activity_labels = read_activity_labels(activity_labels_file)

# Read feature names
feature_names = read_feature_names(features_file)

# Filter feature names to keep only selected features
selected_feature_indices = [feature_names.index(feature) for feature in selected_features]

# Combine data into rows for CSV, including only selected features

train_combined_data = []
train_data_path = 'train-data.csv'

for subject, activity, values in zip(train_subjects, train_activities, train_data):
    values = values.strip().split()
    activity_label = activity_labels[int(activity)]
    selected_values = [values[i] for i in selected_feature_indices]
    row = [subject.strip(), activity_label] + selected_values
    train_combined_data.append(row)

test_combined_data = []
test_data_path= 'test-data.csv'

for subject, activity, values in zip(test_subjects, test_activities, test_data):
    values = values.strip().split()
    activity_label = activity_labels[int(activity)]
    selected_values = [values[i] for i in selected_feature_indices]
    row = [subject.strip(), activity_label] + selected_values
    test_combined_data.append(row)

# Write combined data to CSV file
write_csv(train_data_path, train_combined_data, ['Subject', 'Activity'] + selected_features)
write_csv(test_data_path, test_combined_data, ['Subject', 'Activity'] + selected_features)

print("CSVs file created successfully!")

# Read the CSV file with original column names included
df = pd.read_csv(train_data_path)

# Read the CSV file with original column names included
test_df = pd.read_csv(test_data_path)

# Check for null values and visualize
null_counts = df.isnull().sum()
duplicate_count = df.duplicated().sum()

print(f"Number of null rows \n'{null_counts}' \n\nNumber of duplicate rows: {duplicate_count} \n\n")

# Split the data into features and target
X_train = df[selected_features]
y_train = df['Activity']

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions on the test data
rf_test_predictions = rf_model.predict(test_df[selected_features])

# Compute precision, recall, and F1-score for Random Forest model
rf_precision = precision_score(test_df['Activity'], rf_test_predictions, average='weighted')
rf_recall = recall_score(test_df['Activity'], rf_test_predictions, average='weighted')
rf_f1 = f1_score(test_df['Activity'], rf_test_predictions, average='weighted')

print("\nRandom Forest Model Metrics:")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1-Score: {rf_f1}")

# Train Gradient Boosting Machines model
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_train, y_train)

# Making predictions on the test data
gbm_test_predictions = gbm_model.predict(test_df[selected_features])

# Compute precision, recall, and F1-score for Gradient Boosting Machines model
gbm_precision = precision_score(test_df['Activity'], gbm_test_predictions, average='weighted')
gbm_recall = recall_score(test_df['Activity'], gbm_test_predictions, average='weighted')
gbm_f1 = f1_score(test_df['Activity'], gbm_test_predictions, average='weighted')

print("\nGradient Boosting Machines Model Metrics:")
print(f"Precision: {gbm_precision}")
print(f"Recall: {gbm_recall}")
print(f"F1-Score: {gbm_f1}")

# Exporting results to CSV file
results_df = test_df.copy()
results_df['RF Predicted Activity'] = rf_test_predictions
results_df['GBM Predicted Activity'] = gbm_test_predictions

# Find instances where the algorithms missed the prediction
results_df['RF Missed Prediction'] = results_df['Activity'] != results_df['RF Predicted Activity']
results_df['GBM Missed Prediction'] = results_df['Activity'] != results_df['GBM Predicted Activity']

# Calculate the percentage of correct predictions and missed predictions for each algorithm
rf_correct_percentage = 100 * results_df['RF Missed Prediction'].value_counts(normalize=True)[False]
gbm_correct_percentage = 100 * results_df['GBM Missed Prediction'].value_counts(normalize=True)[False]

# Plotting the bar chart
labels = ['Random Forest', 'Gradient Boosting Machines']
correct_percentages = [rf_correct_percentage, gbm_correct_percentage]
missed_percentages = [100 - rf_correct_percentage, 100 - gbm_correct_percentage]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, correct_percentages, width, label='Correct Prediction')
rects2 = ax.bar(x + width/2, missed_percentages, width, label='Missed Prediction')

ax.set_ylabel('Percentage')
ax.set_title('Correct vs Missed Predictions by Algorithm')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

# Calculate the percentage of missed predictions for each activity
missed_by_activity_rf = results_df[results_df['RF Missed Prediction'] == True]['Activity'].value_counts(normalize=True) * 100
missed_by_activity_gbm = results_df[results_df['GBM Missed Prediction'] == True]['Activity'].value_counts(normalize=True) * 100

# Combine the missed predictions from both algorithms
missed_by_activity = missed_by_activity_rf.add(missed_by_activity_gbm, fill_value=0).sort_values(ascending=False)

# Plotting the bar chart
plt.figure(figsize=(10, 6))
missed_by_activity.plot(kind='bar')
plt.title('Percentage of Missed Predictions by Activity')
plt.xlabel('Activity')
plt.ylabel('Percentage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Calculate the counts of missed predictions for each pair of activities for RF
missed_pairs_rf = results_df[results_df['RF Missed Prediction'] == True][['Activity', 'RF Predicted Activity']].value_counts()

# Calculate the total counts for each correct activity for RF
total_counts_rf = results_df['Activity'].value_counts()

# Calculate the percentage of missed predictions for each pair of activities for RF
percentage_missed_rf = (missed_pairs_rf / total_counts_rf) * 100

# Plotting the bar chart for RF
plt.figure(figsize=(15, 8))
percentage_missed_rf.plot(kind='bar')
plt.title('Percentage of Missed Predictions by Activity Pair (Random Forest)')
plt.xlabel('Activity Pair')
plt.ylabel('Percentage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Calculate the counts of missed predictions for each pair of activities for GBM
missed_pairs_gbm = results_df[results_df['GBM Missed Prediction'] == True][['Activity', 'GBM Predicted Activity']].value_counts()

# Calculate the total counts for each correct activity for GBM
total_counts_gbm = results_df['Activity'].value_counts()

# Calculate the percentage of missed predictions for each pair of activities for GBM
percentage_missed_gbm = (missed_pairs_gbm / total_counts_gbm) * 100

# Plotting the bar chart for GBM
plt.figure(figsize=(15, 8))
percentage_missed_gbm.plot(kind='bar')
plt.title('Percentage of Missed Predictions by Activity Pair (Gradient Boosting Machines)')
plt.xlabel('Activity Pair')
plt.ylabel('Percentage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Exporting results to CSV file
results_df.to_csv('results.csv', index=False)

print("Results exported to 'results.csv' successfully!")