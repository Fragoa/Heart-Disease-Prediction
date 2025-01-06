import numpy as np
import matplotlib.pyplot as plt
import math


def initialize_weights(n_features):
    weights = np.random.randn(n_features) * 0.01
    return weights


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

    
def predict_proba(X, weights):
    linearCombination = np.dot(X,weights)
    probLinearCombination = sigmoid(linearCombination)
    return probLinearCombination

def compute_cost(X, y, weights):
    probLinearCombination = predict_proba(X, weights)  
    loss = y * np.log(probLinearCombination) + (1 - y) * np.log(1 - probLinearCombination)  
    avgLoss = -np.sum(loss) / X.shape[0]  
    return avgLoss


def compute_gradient(X, y, weights):
    probLinearCombination = predict_proba(X, weights)  
    error = probLinearCombination - y  
    gradientCost = (1 / X.shape[0]) * np.dot(X.T, error) 
    
    return gradientCost

def update_weights(weights, gradient, learning_rate):
    adjustment = learning_rate * gradient
    updated_weights = weights - adjustment
    
    return updated_weights
def train(X, y, n_iterations, learning_rate):
    # weights = np.zeros(X.shape[1])
    weights = initialize_weights(X.shape[1])  # X.shape[1] is the number of features (including the bias term)

    costs = []
    
    for i in range(n_iterations):
        
        cost = compute_cost(X, y, weights)
        costs.append(cost)
        gradient = compute_gradient(X, y, weights)
        
        # Update weights
        weights = update_weights(weights, gradient, learning_rate)
        
        # Print cost at intervals (optional)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
    return weights, costs

def predict(X, weights, threshold=0.5):
    probabilities = predict_proba(X, weights)
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions

def evaluate(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def plot_cost(costs):
    plt.plot(costs, label="Cost")
    
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Reduction Over Iterations")
    plt.legend()
    
    plt.show()

def plot_decision_boundary(X, y, weights):
    # Step 1: Create a mesh grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Step 2: Compute predictions for the grid
    grid = np.c_[xx.ravel(), yy.ravel()]  # Combine xx and yy into a grid
    grid = np.hstack([np.ones((grid.shape[0], 1)), grid])  # Add bias term
    probs = predict_proba(grid, weights).reshape(xx.shape)  # Compute probabilities
    
    # Step 3: Plot the decision boundary
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.8)  # Decision boundary at 0.5
    
    # Step 4: Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", s=40, label="Data Points")
    
    # Step 5: Add labels, legend, and title
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.legend(loc="upper right")
    
    # Step 6: Show the plot
    plt.show()