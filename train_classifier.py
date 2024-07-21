#train_classifier.py

import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load augmented data
data_dict = pickle.load(open('augmented_data_single.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train a RandomForest classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
with open('model_single.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model training complete. Model saved to model.p.")
