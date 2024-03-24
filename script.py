import csv
import os
import pandas as pd

base_dir = 'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset'
data_dir = 'train'

subject_file = os.path.join(base_dir, data_dir, 'subject_train.txt')
activity_file = os.path.join(base_dir, data_dir, 'y_train.txt')
data_file = os.path.join(base_dir, data_dir, 'X_train.txt')
activity_labels_file = os.path.join(base_dir, 'activity_labels.txt')

# Define the path to the output CSV file
file_path = 'data.csv'

# Function to read data from a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Function to write data to a CSV file
def write_csv(output_file, rows):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Subject', 'Activity'] + [f'Value_{i}' for i in range(1, 562)])
        writer.writerows(rows)

# Function to parse activity labels
def read_activity_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = {int(line.split()[0]): line.split()[1] for line in lines}
    return labels

# Get data from input files
subjects = read_file(subject_file)
activities = read_file(activity_file)
data = read_file(data_file)

# Parse activity labels
activity_labels = read_activity_labels(activity_labels_file)

# Combine data into rows for CSV
combined_data = []
for subject, activity, values in zip(subjects, activities, data):
    values = values.strip().split()
    activity_label = activity_labels[int(activity)]
    row = [subject.strip(), activity_label] + values
    combined_data.append(row)

# Write combined data to CSV file
write_csv(file_path, combined_data)

print("CSV file created successfully!")

df = pd.read_csv(file_path, header = 0)

# Check for null values and visualize
null_counts = df.isnull().sum()
duplicate_count = df.duplicated().sum()

print(f"Number of null rows \n'{null_counts}' \n\nNumber of duplicate rows: {duplicate_count} \n\n")

# Shuffle the rows to display random ones
df_shuffled = df.sample(frac=1, random_state=42)

print("Top 150 shuffled rows displayed:")
print(df_shuffled.head(150))
