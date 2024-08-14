import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the processed data
file_path = 'processed_train_data.csv'  # Ensure this is the correct path to your processed CSV file
data = pd.read_csv(file_path)

# 2. Ensure the target variable is categorical
target_column = 'ACD110'
if data[target_column].dtype != 'int' and data[target_column].dtype != 'object':
    data[target_column] = pd.Categorical(data[target_column]).codes  # Convert to categorical codes

# 3. Data Splitting
X = data.drop(target_column, axis=1)
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Logistic Regression Model
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)

# 5. Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)

# 6. Model Evaluation Metrics
def save_metrics(y_test, y_pred, y_prob, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Handle ROC AUC for multiclass classification
    if y_prob.shape[1] == len(np.unique(y_test)):
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:
        roc_auc = 'Not computable'  # or handle it differently if needed
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }
    with open(f"{model_name}_metrics.txt", "w") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

save_metrics(y_test, y_pred_logreg, y_prob_logreg, "Logistic_Regression")
save_metrics(y_test, y_pred_rf, y_prob_rf, "Random_Forest")

# 7. Plotting Figures

# ROC Curve
plt.figure(figsize=(10, 6))
n_classes = len(np.unique(y_test))
for i in range(n_classes):
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test == i, y_prob_logreg[:, i])
    plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (Class {i})')
    fpr_rf, tpr_rf, _ = roc_curve(y_test == i, y_prob_rf[:, i])
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (Class {i})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# Confusion Matrix for Logistic Regression
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix_logreg, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig("conf_matrix_logreg.png")
plt.close()

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Greens")
plt.title('Confusion Matrix - Random Forest')
plt.savefig("conf_matrix_rf.png")
plt.close()

# Feature Importance for Random Forest
feature_importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color="green")
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest')
plt.savefig("feature_importance_rf.png")
plt.close()
