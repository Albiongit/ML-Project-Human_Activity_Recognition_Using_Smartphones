import csv
from logging import root
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from tkinter import Scrollbar, Tk, filedialog, Listbox, Button, Label, END, Frame, Canvas
from tkinter.messagebox import showinfo
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.svm import SVC
from tkinter.messagebox import showinfo
from tkinter import filedialog, messagebox


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
test_data_path = 'test-data.csv'

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

# Train SVM model
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Making predictions on the test data
svm_test_predictions = svm_model.predict(test_df[selected_features])

# Compute precision, recall, and F1-score for SVM model
svm_precision = precision_score(test_df['Activity'], svm_test_predictions, average='weighted')
svm_recall = recall_score(test_df['Activity'], svm_test_predictions, average='weighted')
svm_f1 = f1_score(test_df['Activity'], svm_test_predictions, average='weighted')

print("\nSupport Vector Machine Model Metrics:")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}")
print(f"F1-Score: {svm_f1}")

# Exporting results to CSV file
results_df = test_df.copy()
results_df['RF Predicted Activity'] = rf_test_predictions
results_df['GBM Predicted Activity'] = gbm_test_predictions
results_df['SVM Predicted Activity'] = svm_test_predictions

# Find instances where the algorithms missed the prediction
results_df['RF Missed Prediction'] = results_df['Activity'] != results_df['RF Predicted Activity']
results_df['GBM Missed Prediction'] = results_df['Activity'] != results_df['GBM Predicted Activity']
results_df['SVM Missed Prediction'] = results_df['Activity'] != results_df['SVM Predicted Activity']

# Calculate the percentage of correct predictions and missed predictions for each algorithm
rf_correct_percentage = 100 * results_df['RF Missed Prediction'].value_counts(normalize=True)[False]
gbm_correct_percentage = 100 * results_df['GBM Missed Prediction'].value_counts(normalize=True)[False]
svm_correct_percentage = 100 * results_df['SVM Missed Prediction'].value_counts(normalize=True)[False]

# Plotting the bar chart for correct predictions
labels = ['Random Forest', 'Gradient Boosting Machines', 'Support Vector Machine']
correct_percentages = [rf_correct_percentage, gbm_correct_percentage, svm_correct_percentage]
missed_percentages = [100 - rf_correct_percentage, 100 - gbm_correct_percentage, 100 - svm_correct_percentage]

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
missed_by_activity_svm = results_df[results_df['SVM Missed Prediction'] == True]['Activity'].value_counts(normalize=True) * 100

# Combine the missed predictions from both algorithms
missed_by_activity = missed_by_activity_rf.add(missed_by_activity_gbm, fill_value=0).add(missed_by_activity_svm, fill_value=0).sort_values(ascending=False)

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

# Calculate the counts of missed predictions for each pair of activities for SVM
missed_pairs_svm = results_df[results_df['SVM Missed Prediction'] == True][['Activity', 'SVM Predicted Activity']].value_counts()

# Calculate the total counts for each correct activity for SVM
total_counts_svm = results_df['Activity'].value_counts()

# Calculate the percentage of missed predictions for each pair of activities for SVM
percentage_missed_svm = (missed_pairs_svm / total_counts_svm) * 100

# Plotting the bar chart for SVM
plt.figure(figsize=(15, 8))
percentage_missed_svm.plot(kind='bar')
plt.title('Percentage of Missed Predictions by Activity Pair (Support Vector Machine)')
plt.xlabel('Activity Pair')
plt.ylabel('Percentage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Exporting results to CSV file
results_df.to_csv('results.csv', index=False)

print("Results exported to 'results.csv' successfully!")

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    messagebox.askyesno("Contribute Data", "Do you want your data to contribute to our model training?")

    uploaded_df = pd.read_csv(file_path)
    uploaded_df['Predicted Activity'] = svm_model.predict(uploaded_df[selected_features])
    uploaded_df.to_csv('uploaded_with_predictions.csv', index=False)

    global subjects
    subjects = uploaded_df['Subject'].unique()

    subject_canvas.delete('all')
    icon = Image.open('person-icon.png')
    icon = icon.resize((30, 30))
    icon = ImageTk.PhotoImage(icon)
    subject_canvas.image = icon 

    rows, cols = (len(subjects) + 4) // 5, 5
    for i, subject in enumerate(subjects):
        row, col = divmod(i, cols)
        x, y = col * 100 + 50, row * 100 + 50
        item_id = subject_canvas.create_image(x, y, image=icon, tags=str(subject))
        subject_canvas.create_text(x, y + 40, text=str(subject), tags=str(subject))
        subject_canvas.tag_bind(item_id, "<Button-1>", show_chart)

    subject_canvas.update()
    messagebox.showinfo("File Upload", "File uploaded and predictions made successfully!")

def show_general_statistics():
    data = pd.read_csv('uploaded_with_predictions.csv')
    activity_counts = data['Predicted Activity'].value_counts(normalize=True) * 100

    def plot_general_distribution():
        plt.figure(figsize=(8, 8))
        plt.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
        plt.title('Percentage of Activity for All Subjects')
        plt.gca().add_artist(plt.Circle((0, 0), 0.6, color='white'))
        plt.axis('equal')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def plot_activity_distribution(selected_activity):
        activity_data = data[data['Predicted Activity'] == selected_activity]
        subject_counts = activity_data['Subject'].value_counts(normalize=True) * 100

        plt.figure(figsize=(8, 8))
        subject_counts.sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Subjects Distribution for Activity: {selected_activity}')
        plt.xlabel('Subject')
        plt.ylabel('Percentage')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(plt.gcf(), master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def on_activity_select(event):
        chart_frame.pack_forget()
        chart_frame.pack(side="bottom", fill="both", expand=True)
        selected_activity = activity_listbox.get(activity_listbox.curselection())
        for widget in chart_frame.winfo_children():
            widget.destroy()
        if selected_activity == "General Distribution":
            plot_general_distribution()
        else:
            plot_activity_distribution(selected_activity)

    root = Tk()
    root.title("Activity Distribution")

    general_frame = Frame(root)
    general_frame.pack()

    chart_frame = Frame(root)
    chart_frame.pack(side="bottom", fill="both", expand=True)

    activity_listbox = Listbox(general_frame, selectmode="single", width=50)
    activity_listbox.insert(END, "General Distribution")
    activity_list = data['Predicted Activity'].unique()
    for activity in activity_list:
        activity_listbox.insert(END, activity)
    activity_listbox.pack(side="left", fill="both", expand=True)

    activity_listbox.bind("<<ListboxSelect>>", on_activity_select)

    plot_general_distribution()

    root.mainloop()

def show_chart(event):
    selected_subject = event.widget.gettags("current")[0] 
    subject_data = pd.read_csv('uploaded_with_predictions.csv')
    subject_data = subject_data[subject_data['Subject'] == int(selected_subject)]

    activity_counts = subject_data['Predicted Activity'].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
    plt.title(f'Activity Distribution for Subject {selected_subject}')
    plt.show()

def main():
    global subject_canvas

    root = Tk()
    root.title("Activity Recognition")
    root.geometry("800x600") 
    root.resizable(False, False)  
    window_width = root.winfo_reqwidth()
    window_height = root.winfo_reqheight()
    position_right = int(root.winfo_screenwidth() / 2 - window_width / 2)
    position_down = int(root.winfo_screenheight() / 2 - window_height / 2)
    root.geometry("+{}+{}".format(position_right, position_down))

    frame = Frame(root)
    frame.pack(pady=20)

    title_label = Label(frame, text="Activity Recognition", font=("Helvetica", 16, "bold"))
    title_label.pack()

    subtitle_label = Label(frame, text="Upload your data and view activity predicted by subject", font=("Helvetica", 12))
    subtitle_label.pack()

    upload_button = Button(root, text="Upload your data", command=upload_file, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
    upload_button.pack(pady=20)

    general_statistics_button = Button(root, text="More statistics", command=show_general_statistics, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
    general_statistics_button.pack(pady=10)

    subject_frame = Frame(root)
    subject_frame.pack()

    canvas = Canvas(subject_frame, width=600, height=400)
    canvas.pack(side="left")

    scrollbar = Scrollbar(subject_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    subject_canvas = Canvas(canvas, width=600, height=400)
    canvas.create_window((0, 0), window=subject_canvas, anchor="nw")
    subject_canvas.bind("<Button-1>", show_chart)

    root.mainloop()

if __name__ == "__main__":
    main()
