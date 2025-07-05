import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR example)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Output labels for XOR
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Set random seed for reproducibility
np.random.seed(42)

# Network architecture
input_layer_neurons = X.shape[1]    # 2 input features
hidden_layer_neurons = 4            # Number of hidden layer neurons
output_neurons = 1                  # Binary classification output

# Weight and bias initialization
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))  # input -> hidden
bh = np.random.uniform(size=(1, hidden_layer_neurons))

wo = np.random.uniform(size=(hidden_layer_neurons, output_neurons))       # hidden -> output
bo = np.random.uniform(size=(1, output_neurons))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    predicted_output = sigmoid(final_input)

    # Error calculation
    error = y - predicted_output

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_output.dot(wo.T)
    d_hidden = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Update weights and biases
    wo += hidden_output.T.dot(d_output) * learning_rate
    bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Optional: print error every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# Final output
print("\nFinal predicted output:")
print(np.round(predicted_output, 3))
