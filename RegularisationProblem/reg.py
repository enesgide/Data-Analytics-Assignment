import numpy as np
import matplotlib.pyplot as plt
import funcs

# Load data
train = np.load('train.npy')
test = np.load('test.npy')

# Get x, y values
x_train = train[:, 0]
y_train = train[:, 1]
x_test = test[:, 0]
y_test = test[:, 1]

# Define constant variables
degree = 7
lamb = 2

# Calculate theta hat
A_train = funcs.vandermonde(x_train, degree + 1)
A_tilde = np.concatenate((np.zeros((degree, 1)), np.sqrt(lamb) * np.eye(degree)), axis=1)
A_tilde = np.concatenate((A_train, A_tilde))
y_tilde = np.concatenate((y_train, np.zeros(degree)))
A_tilde_pinv = np.linalg.pinv(A_tilde)
theta_hat = A_tilde_pinv @ y_tilde

# Calculate plotting data
x_plotting = np.linspace(-2.5, 3, 50)
A_plotting = funcs.vandermonde(x_plotting, degree + 1)
y_plotting = A_plotting @ theta_hat

# Visualise data
plt.scatter(x_train, y_train, label='Training data')
plt.scatter(x_test, y_test, label='Test data')
plt.plot(x_plotting, y_plotting, label='Regularised model')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.savefig('plot.png')
