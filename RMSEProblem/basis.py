import numpy as np
import matplotlib.pyplot as plt
import funcs

# Load data
training = np.load('temperature_train.npy')
test = np.load('temperature_test.npy')

# Create time vectors
t_training = np.arange(1, training.shape[0] + 1)
t_test = np.arange(training.shape[0] + 1, training.shape[0] + test.shape[0] + 1)

# Generate omegas and store corresponding RMS errors
omegas = np.linspace(0, 0.5, 100)
training_errors = np.zeros(omegas.shape[0])
test_errors = np.zeros(omegas.shape[0])

for i in range(omegas.shape[0]):
    omega = omegas[i]

    # Calculate y_d_hats
    y_d_hat_training, theta_hat = funcs.y_d_hat(training, test, t_training, omega, None)
    y_d_hat_test, theta_hat = funcs.y_d_hat(training, test, t_test, omega, theta_hat)

    # Calculate the RMS error
    RMSE_train = funcs.RMS_error(training, y_d_hat_training)
    RMSE_test = funcs.RMS_error(test, y_d_hat_test)

    # Store RMS errors
    training_errors[i] = RMSE_train
    test_errors[i] = RMSE_test

# Plot the results
plt.plot(omegas, training_errors, label='Training data')
plt.plot(omegas, test_errors, label='Test data')

plt.xlabel('Omega')
plt.ylabel('RMS error')
plt.legend()

plt.savefig('plot.png')

# Omega corresponding with minimum test RMSE
min_err_i = np.argmin(test_errors)
print(min_err_i)
min_omega = omegas[min_err_i]
print(min_omega.round(4))
