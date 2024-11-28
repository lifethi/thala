import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
df = pd.read_excel('Online Retail.xlsx')

# Cleaning the dataset (drop duplicates and handle missing values)
df = df.drop_duplicates()
df = df.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'Quantity'])

# Optional: Filter data to reduce dataset size
df = df[df['Quantity'] > 0]  # Keep only positive quantities
df = df[df['Country'] == 'United Kingdom']  # Focus on a specific region (optional)

# Remove any duplicate InvoiceDate columns if present
df = df.loc[:, ~df.columns.duplicated()]

# Create a basket format (pivot the data to represent transactions)
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Convert the quantity data into binary (if purchased or not)
def encode_units(x):
    return x > 0  # Returns a boolean

basket = basket.applymap(encode_units)

# Ensure basket is boolean for optimal memory usage
basket = basket.astype('bool')

# Apply Apriori algorithm with a higher minimum support to reduce memory usage
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
df = pd.read_excel('Online Retail.xlsx')

# Cleaning the dataset (drop duplicates and handle missing values)
df = df.drop_duplicates()
df = df.dropna(subset=['InvoiceNo', 'StockCode', 'Description', 'Quantity'])

# Optional: Filter data to reduce dataset size
df = df[df['Quantity'] > 0]  # Keep only positive quantities
df = df[df['Country'] == 'United Kingdom']  # Focus on a specific region (optional)

# Remove any duplicate InvoiceDate columns if present
df = df.loc[:, ~df.columns.duplicated()]

# Create a basket format (pivot the data to represent transactions)
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Convert the quantity data into binary (if purchased or not)
def encode_units(x):
    return x > 0  # Returns a boolean

basket = basket.applymap(encode_units)

# Ensure basket is boolean for optimal memory usage
basket = basket.astype('bool')

# Apply Apriori algorithm with a higher minimum support to reduce memory usage
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

# Generate association rules with a lift metric and a minimum threshold
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Print the frequent itemsets and the association rules
print("Frequent Itemsets:")
print(frequent_itemsets.head())  # Show only the first few rows
print("\nAssociation Rules:")
print(rules.head())  # Show only the first few rows

# Save the results to a file
frequent_itemsets.to_excel('frequent_itemsets.xlsx', index=False)
rules.to_excel('association_rules.xlsx', index=False)
# Generate association rules with a lift metric and a minimum threshold
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Print the frequent itemsets and the association rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)
