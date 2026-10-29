from pathlib import Path
import os

from preprocessing import (
    num_encoder_tokens,
    num_decoder_tokens,
    encoder_input_data,
    decoder_input_data,
    decoder_target_data
)

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

latent_dim = 256
batch_size = 50
epochs = 100

encoder_inputs = Input(
    shape=(None, num_encoder_tokens)
)

encoder_lstm = LSTM(
    latent_dim,
    return_state=True
)

encoder_outputs, state_hidden, state_cell = encoder_lstm(
    encoder_inputs
)

encoder_states = [state_hidden, state_cell]

decoder_inputs = Input(
    shape=(None, num_decoder_tokens)
)

decoder_lstm = LSTM(
    latent_dim,
    return_sequences=True,
    return_state=True
)

decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(
    decoder_inputs,
    initial_state=encoder_states
)

decoder_dense = Dense(
    num_decoder_tokens,
    activation="softmax"
)

decoder_outputs = decoder_dense(
    decoder_outputs
)

training_model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

print("\nModel Summary\n")
training_model.summary()

training_model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nTraining Model...\n")

training_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping]
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "training_model.keras"

training_model.save(MODEL_PATH)

print(f"\nModel saved to:\n{MODEL_PATH}")