import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'D:/SEM 5/dwdm lab/apriori/Online Retail.xlsx'  # replace with your file path
xls = pd.ExcelFile(file_path)

# Load the data
df = pd.read_excel(xls, sheet_name='Online Retail')

# Step 1: Clean the data by removing rows with missing CustomerID
df_clean = df.dropna(subset=['CustomerID']).copy()

# Step 2: Create a new column "TotalAmountSpent" (Quantity * UnitPrice)
df_clean.loc[:, 'TotalAmountSpent'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Step 3: Group by CustomerID to calculate the total amount spent by each customer
customer_data = df_clean.groupby('CustomerID').agg({
    'TotalAmountSpent': 'sum'
}).reset_index()

# Step 4: Scale the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['TotalAmountSpent']])

# Step 5: Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# Step 6: Calculate the Silhouette Score
sil_score = silhouette_score(customer_data_scaled, customer_data['Cluster'])
print(f"Silhouette Score: {sil_score}")

# Step 7: Visualize the clusters
plt.scatter(customer_data['CustomerID'], customer_data['TotalAmountSpent'], c=customer_data['Cluster'], cmap='viridis')
plt.xlabel('CustomerID')
plt.ylabel('TotalAmountSpent')
plt.title('Customer Clusters based on Spending')
plt.show()

# Step 8: Analyze the cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)
