import numpy as np

def y_d_hat(training, test, t, omega, theta_hat):
    # Reshape the training time vector
    t = t.reshape(-1, 1)

    # Apply the basis functions
    col1 = np.ones(t.shape[0]).reshape(-1, 1)
    col2 = t.reshape(-1, 1)
    col3 = np.cos(omega * t).reshape(-1, 1)

    # Get A_training and the pseudo inverse
    A = np.concatenate((col1,col2,col3), axis=1)
    A_pinv = np.linalg.pinv(A)

    # Calculate theta_hat and y_d_hat
    if theta_hat is None:
        theta_hat = A_pinv @ training
    y_d_hat = A @ theta_hat

    return y_d_hat, theta_hat

def RMS_error(y_d, y_d_hat):
    num = np.linalg.norm(y_d_hat - y_d)
    den = np.sqrt(y_d.shape[0])
    return num / den