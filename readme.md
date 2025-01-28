# Stroke Analysis Project

## Project Overview

This project analyzes a dataset related to stroke occurrences using various machine learning techniques and visualizations. The primary goals include:

- Preprocessing data to prepare it for machine learning models.
- Building and evaluating models such as Decision Trees, K-Means clustering, and Artificial Neural Networks (ANN).
- Generating and visualizing association rules.

## Features

1. **Data Preprocessing**:

   - Encode categorical variables into numerical values.
   - Save the preprocessed dataset for further analysis.

2. **Machine Learning Models**:

   - **Decision Tree**: Train a classifier to predict stroke occurrences and visualize it.
   - **K-Means Clustering**: Apply clustering to group data into clusters.
   - **Artificial Neural Network (ANN)**: Train and visualize an ANN model for stroke prediction.

3. **Association Rules**:
   - Generate frequent itemsets using the Apriori algorithm.
   - Visualize association rules based on support, confidence, and lift.

## Folder Structure

```
python assignment/
├── data/
│   ├── stroke_data.csv                # Original dataset
│   ├── preprocessed_stroke_data.csv   # Preprocessed dataset
├── models/
│   ├── decision_tree.py               # Code for Decision Tree model
│   ├── kmeans_clustering.py           # Code for K-Means Clustering
│   ├── association_rules.py           # Code for generating Association Rules
│   ├── ann.py                         # Code for Artificial Neural Network
├── main.py                            # Main script orchestrating the project workflow
├── README.md                          # Documentation for the project
```

## Project Workflow

### 1. Data Preprocessing

- Load the dataset from `stroke_data.csv`.
- Encode categorical variables (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`).
- Save the preprocessed data to `preprocessed_stroke_data.csv`.

### 2. Decision Tree

- Train a Decision Tree classifier to predict strokes.
- Evaluate the model using accuracy and a confusion matrix.
- Save the Decision Tree visualization (`decision_tree.png`) and Confusion Matrix (`confusion_matrix.png`).

### 3. K-Means Clustering

- Standardize numerical features (`age`, `avg_glucose_level`, `bmi`).
- Apply K-Means clustering with 3 clusters.
- Reduce dimensions using PCA for visualization.
- Save the clustering visualization (`kmeans_clustering.png`).

### 4. Association Rules

- Convert the dataset into binary format.
- Generate frequent itemsets using the Apriori algorithm.
- Visualize:
  - Support vs Confidence (`support_vs_confidence.png`).
  - Support vs Lift (`support_vs_lift.png`).

### 5. Artificial Neural Network (ANN)

- Train an ANN with input, hidden, and output layers to predict strokes.
- Evaluate the model using accuracy and loss over epochs.
- Save the accuracy (`ann_accuracy.png`) and loss (`ann_loss.png`) plots.

## Main Script

The `main.py` script orchestrates the entire workflow:

```python
if __name__ == "__main__":
    data = load_and_preprocess_data()
    run_decision_tree(data)
    run_kmeans(data)
    run_association_rules(data)
    run_ann(data)
```

## Dependencies

To run the project, ensure you have the following Python libraries installed:

- pandas
- matplotlib
- scikit-learn
- keras
- mlxtend

Install the dependencies using:

```bash
pip install pandas matplotlib scikit-learn keras mlxtend
```

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project folder:
   ```bash
   cd project-root
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Outputs

Upon running the script, the following outputs will be generated:

- Decision Tree and Confusion Matrix visualizations.
- K-Means clustering visualization.
- Association Rules visualizations (Support vs Confidence, Support vs Lift).
- ANN accuracy and loss plots.

---

Happy analyzing! 🎉

# contact me - [! prashantadeuja@gmail.com]
