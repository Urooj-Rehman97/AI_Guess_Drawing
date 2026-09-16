from sklearn.tree import DecisionTreeClassifier

# Training data
X = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]

# Answers
y = [
    "Star",
    "Star",
    "Heart",
    "Heart"
]

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X, y)

# New data
new_data = [[1, 1]]

# Prediction
prediction = model.predict(new_data)

print(prediction)