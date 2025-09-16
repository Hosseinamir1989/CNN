import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the model
model = load_model("setup_number1/cnn_model_AAPL.h5")

# Load or recreate your X_test, y_test and scaler_y
# These should match what you used when saving the model
# e.g., from a saved .npz or pickle file
# X_test.shape → (n_samples, window_size, n_features)
# y_test.shape → (n_samples,)

# Predict
y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Evaluate
rmse = mean_squared_error(y_test_orig, y_pred, squared=False)
mae = mean_absolute_error(y_test_orig, y_pred)
diracc = np.mean(np.sign(y_test_orig[1:] - y_test_orig[:-1]) ==
                 np.sign(y_pred[1:] - y_pred[:-1]))

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Directional Accuracy: {diracc:.4f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(y_test_orig, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linestyle="--")
plt.title("Actual vs Predicted (CNN Model for AAPL)")
plt.xlabel("Test Sample")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
