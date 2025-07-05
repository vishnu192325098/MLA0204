import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])
np.random.seed(42)
input_layer_neurons = X.shape[1]    
hidden_layer_neurons = 4            
output_neurons = 1                  
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons)) 
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wo = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))
learning_rate = 0.1
epochs = 10000
for epoch in range(epochs):
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, wo) + bo
    predicted_output = sigmoid(final_input)
    error = y - predicted_output
    d_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_output.dot(wo.T)
    d_hidden = error_hidden_layer * sigmoid_derivative(hidden_output)
    wo += hidden_output.T.dot(d_output) * learning_rate
    bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch} | Loss: {loss:.4f}")
print("\nFinal predicted output:")
print(np.round(predicted_output, 3))
