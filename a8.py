from neural import *

# print("Training XOR Network with 2 Hidden Nodes\n")

# xor_training_data = [([0, 0], [0]),
#                      ([0, 1], [1]),
#                      ([1, 0], [1]),
#                      ([1, 1], [0])]

# # Create the network
# nn = NeuralNet(2, 8, 1)

# # Train the network
# nn.train(xor_training_data, iters=5000, print_interval=100)

# # Test the network after training
# results = nn.test_with_expected(xor_training_data)
# for input_vals, expected_output, actual_output in results:
#     print(f"Input: {input_vals}, Expected: {expected_output}, Actual: {actual_output}")

# print("\nTraining Voter Opinion Network\n")

# # Training data from Table 2
# voter_training_data = [
#     ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),  # Republican
#     ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),  # Republican
#     ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),  # Republican
#     ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),  # Democrat
#     ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),  # Democrat
#     ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])   # Democrat
# ]

# # Test data from Table 3
# voter_test_data = [
#     ([1.0, 1.0, 1.0, 0.1, 0.1]),
#     ([0.5, 0.2, 0.1, 0.7, 0.7]),
#     ([0.8, 0.3, 0.3, 0.3, 0.8]),
#     ([0.8, 0.3, 0.3, 0.8, 0.3]),
#     ([0.9, 0.8, 0.8, 0.3, 0.6])
# ]

# # Create and train network
# voter_nn = NeuralNet(5, 3, 1)
# voter_nn.train(voter_training_data, iters=5000, print_interval=500)

# # Test on new voters
# for voter in voter_test_data:
#     prediction = voter_nn.evaluate(voter)
#     print(f"Input: {voter}, Predicted Party: {round(prediction[0])}")

# Housing prices
import csv
import random

print("\nTraining House Price Prediction Network\n")

# Step 1: Load and parse data from output.csv
with open("output.csv", "r") as f:
    reader = csv.DictReader(f)
    raw_data = list(reader)

# Step 2: Extract features and prices
features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "sqft_above", "sqft_basement",
            "yr_built", "yr_renovated"]

house_data = []
for row in raw_data:
    try:
        x = [float(row[feat]) for feat in features]
        y = [float(row["price"])]
        house_data.append((x, y))
    except:
        continue  # skip rows with missing/invalid data

# Step 3: Normalize inputs
def normalize(dataset):
    num_features = len(dataset[0][0])
    mins = [float('inf')] * num_features
    maxes = [float('-inf')] * num_features

    for inputs, _ in dataset:
        for i, val in enumerate(inputs):
            mins[i] = min(mins[i], val)
            maxes[i] = max(maxes[i], val)

    for inputs, _ in dataset:
        for i in range(len(inputs)):
            if maxes[i] - mins[i] != 0:
                inputs[i] = (inputs[i] - mins[i]) / (maxes[i] - mins[i])
            else:
                inputs[i] = 0.0
    return dataset

house_data = normalize(house_data)

# Step 4: Split into training and test sets
random.shuffle(house_data)
split_idx = int(0.8 * len(house_data))
train_data = house_data[:split_idx]
test_data = house_data[split_idx:]

# Step 5: Train the neural network
nn = NeuralNet(len(features), 10, 1)
nn.train(train_data, iters=10000, print_interval=1000)

# Step 6: Test predictions and calculate margin of error
print("\nPredictions vs. Actual Prices:\n")
for inputs, expected in test_data[:25]:  # limit to first 25 for readability
    predicted = nn.evaluate(inputs)[0]
    print(f"Actual: ${expected[0]:,.2f}, Predicted: ${predicted:,.2f}, Error: ${abs(expected[0] - predicted):,.2f}")


