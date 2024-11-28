# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = r'D:/SEM 5/dwdm lab/apriori/E-commerce Customer Behavior - Sheet1.csv'  # raw string for the file path
df = pd.read_csv(file_path)

# Drop rows with missing values (if any)
df = df.dropna()

# Inspect the column names in the dataset
print("Columns in dataset:", df.columns)

# Select features (X) and target (y) for classification
# In this case, we are assuming that the target variable is 'Satisfaction Level'
X = df[['Total Spend', 'Items Purchased', 'Age', 'Discount Applied', 'Days Since Last Purchase']]
y = df['Satisfaction Level']  # Target: 'Satisfaction Level'

# Encode the categorical target variable 'Satisfaction Level' (if it's not already numeric)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode 'Satisfaction Level'

# Handle categorical columns in X (if any), e.g., 'Discount Applied'
if df['Discount Applied'].dtype == 'object':
    df['Discount Applied'] = label_encoder.fit_transform(df['Discount Applied'])

# Standardize the feature columns (X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create and train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
