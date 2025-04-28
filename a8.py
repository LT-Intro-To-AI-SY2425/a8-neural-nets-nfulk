from neural import *

print("Training XOR Network with 2 Hidden Nodes\n")

xor_training_data = [([0, 0], [0]),
                     ([0, 1], [1]),
                     ([1, 0], [1]),
                     ([1, 1], [0])]

# Create the network
nn = NeuralNet(2, 8, 1)

# Train the network
nn.train(xor_training_data, iters=5000, print_interval=100)

# Test the network after training
results = nn.test_with_expected(xor_training_data)
for input_vals, expected_output, actual_output in results:
    print(f"Input: {input_vals}, Expected: {expected_output}, Actual: {actual_output}")

print("\nTraining Voter Opinion Network\n")

# Training data from Table 2
voter_training_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),  # Republican
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),  # Republican
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),  # Republican
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),  # Democrat
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),  # Democrat
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])   # Democrat
]

# Test data from Table 3
voter_test_data = [
    ([1.0, 1.0, 1.0, 0.1, 0.1]),
    ([0.5, 0.2, 0.1, 0.7, 0.7]),
    ([0.8, 0.3, 0.3, 0.3, 0.8]),
    ([0.8, 0.3, 0.3, 0.8, 0.3]),
    ([0.9, 0.8, 0.8, 0.3, 0.6])
]

# Create and train network
voter_nn = NeuralNet(5, 3, 1)
voter_nn.train(voter_training_data, iters=5000, print_interval=500)

# Test on new voters
for voter in voter_test_data:
    prediction = voter_nn.evaluate(voter)
    print(f"Input: {voter}, Predicted Party: {round(prediction[0])}")

