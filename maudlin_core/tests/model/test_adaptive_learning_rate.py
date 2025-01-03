import pytest
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from maudlin_core.src.model.adaptive_learning_rate import AdaptiveLearningRate  # Assuming this callback is saved in 'adaptive_lr.py'

@pytest.fixture
def model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(10,)),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

@pytest.fixture
def data():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def callback(model):
    callback = AdaptiveLearningRate(metric_name='loss', patience=2, factor=0.5, min_lr=1e-6)
    callback.set_model(model)  # Attach the model to the callback
    return callback

def test_reduce_lr_on_plateau(model, data, callback):
    X, y = data

    # Simulate training with constant loss
    logs_list = [{'loss': 0.5}, {'loss': 0.5}, {'loss': 0.5}, {'loss': 0.5}]

    # Attach callback to the model
    model.fit(X, y, epochs=1, callbacks=[callback], verbose=0)

    # Mock the epoch logs to simulate no improvement
    for epoch, logs in enumerate(logs_list):
        callback.on_epoch_end(epoch, logs)

    # Check that learning rate was reduced
    optimizer = model.optimizer
    old_lr = 0.01
    expected_lr = old_lr * 0.5 * 0.5  # Reduced twice due to patience
    current_lr = float(optimizer.learning_rate.numpy())

    assert pytest.approx(current_lr, rel=1e-5) == expected_lr, f"Expected LR: {expected_lr}, but got: {current_lr}"

def test_no_lr_below_min_lr(model, data, callback):
    X, y = data

    # Set learning rate close to min_lr
    optimizer = model.optimizer
    optimizer.learning_rate.assign(1e-6)

    # Trigger a reduction attempt
    callback.on_epoch_end(1, {'loss': 0.6})
    callback.on_epoch_end(2, {'loss': 0.6})
    callback.on_epoch_end(3, {'loss': 0.6})

    # Check that learning rate did not go below min_lr
    current_lr = float(optimizer.learning_rate.numpy())
    assert pytest.approx(current_lr, rel=1e-5) == 1e-6, f"Learning rate should not go below 1e-6, but got: {current_lr}"

