#!/bin/bash

# Check if a single argument (path to data file) is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_data_file>"
    exit 1
fi

data_path="$1"

# Check if the provided file exists
if [ ! -f "$data_path" ]; then
    echo "Error: File '$data_path' not found"
    exit 1
fi

# Python script to evaluate KNN using the provided data file
python_script=$(cat <<EOF
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import Counter
from sklearn.model_selection import train_test_split

def euclidean(point1, point2):
    if len(point1) != len(point2):
        raise ValueError('Lengths do not match in distance calculation')
    return np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(point1, point2)]))

def manhattan(point1, point2):
    if len(point1) != len(point2):
        raise ValueError('Lengths do not match in distance calculation')
    return np.sum([abs(x - y) for x, y in zip(point1, point2)])

def chebyshev(point1, point2):
    if(len(point1) != len(point2)):
        raise ValueError('Lengths do not match in distance calculation')
    return np.max([abs(x-y) for x, y in zip(point1, point2)])


from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean', encoding_type='resnet'):
        self.k = k
        self.distance_metric = distance_metric
        self.encoding_type = encoding_type

    def _get_distance(self, point1, point2):
        if self.distance_metric == 'euclidean':
            return euclidean(point1, point2)
        elif self.distance_metric == 'manhattan':
            return manhattan(point1, point2)
        elif self.distance_metric == 'chebyshev':
            return chebyshev(point1, point2)
        else:
            raise ValueError("Invalid distance metric")

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def prediction(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = [self._get_distance(test_point, train_point) for train_point in self.X_train]
            sorted_indices = np.argsort(distances)
            k_nearest_indices = sorted_indices[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
            
            label_counter = Counter(k_nearest_labels)
            most_common_label = label_counter.most_common(1)[0][0]
            predictions.append(most_common_label)
        return predictions


if len(sys.argv) != 2:
    print("Usage: python knn_eval.py <path_to_data_file>")
    sys.exit(1)

data_path = sys.argv[1]
dataset = np.load(data_path, allow_pickle=True)

label_names = dataset[:, 3]
names = ['Game ID', 'ResNet Embeddings', 'VIT Embeddings', 'Label Name', 'Guess']
unique_labels, label_count = np.unique(label_names, return_counts=True)

# Create a list to hold the results for ResNet and VIT embeddings
results = []

for encoding_type in ['ResNet', 'VIT']:
    # Use either ResNet or VIT embeddings based on the encoding type
    if encoding_type == 'ResNet':
        embeddings = dataset[:, 1]  # ResNet Embeddings (column index 1)
    elif encoding_type == 'VIT':
        embeddings = dataset[:, 2]  # VIT Embeddings (column index 2)
    
    # Split data into training and validation subsets
    X_train, X_val, y_train, y_val = train_test_split(embeddings, label_names, test_size=0.3)

    # Create KNN classifier instance
    knn = KNNClassifier(k=5, distance_metric='euclidean', encoding_type=encoding_type)

    # Fit the classifier on training data
    knn.fit(X_train, y_train)

    # Predict on validation data
    y_pred = knn.prediction(X_val)

    # Evaluate the classifier's performance
    f1 = f1_score(y_val, y_pred, average='weighted')
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=1)  # Add zero_division parameter, precision for a class will be set to 0 when there are no predicted samples for that class, and it won't result in a warning. 
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=1)

    # Append results to the list
    results.append((encoding_type, accuracy, precision, recall, f1))


# Print the results
print("Distance Metric = Euclidean")
print(f"{'Encoding Type':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
for result in results:
    encoding_type, accuracy, precision, recall, f1 = result
    print(f"{encoding_type:<15} {accuracy:.4f} {precision:.4f} {recall:.4f} {f1:.4f}")
EOF
)

# Run the Python script
echo "$python_script" | python - "$data_path"
