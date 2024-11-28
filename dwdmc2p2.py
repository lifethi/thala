# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = r'D:\SEM 5\dwdm lab\apriori\E-commerce Customer Behavior - Sheet1.csv'  # raw string for the file path
df = pd.read_csv(file_path)

# Preprocessing - Encoding categorical features using LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['City'] = label_encoder.fit_transform(df['City'])
df['Membership Type'] = label_encoder.fit_transform(df['Membership Type'])
df['Discount Applied'] = df['Discount Applied'].apply(lambda x: 1 if x == 'TRUE' else 0)

# Separating features and target
X = df.drop('Satisfaction Level', axis=1)  # Features
y = df['Satisfaction Level']  # Target (Satisfaction Level)
y = label_encoder.fit_transform(y)  # Encoding the target variable

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Visualize the Decision Tree with limited depth for better readability
plt.figure(figsize=(25, 15))  # Increase figure size to improve clarity
plot_tree(clf, 
          filled=True, 
          feature_names=X.columns, 
          class_names=label_encoder.classes_, 
          max_depth=3,  # Limit tree depth for readability
          fontsize=12)  # Adjusting font size for readability
plt.title('Decision Tree')
plt.show()  # Ensure the decision tree plot is shown

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))  # Adjust figure size for better clarity
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()  # Ensure the confusion matrix plot is shown
