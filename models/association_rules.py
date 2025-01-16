import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Load preprocessed data
data = pd.read_csv('data/preprocessed_stroke_data.csv')

# Convert to boolean DataFrame
data_bool = data.astype(bool)

# Apply Apriori algorithm
frequent_itemsets = apriori(data_bool, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Print rules
print(rules)

# Plot support vs confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()

# Plot support vs lift
plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Lift')
plt.title('Support vs Lift')
plt.show()