from pathlib import Path
import re
import numpy as np
import streamlit as st

from keras.layers import Input
from keras.models import Model, load_model

from src.preprocessing import (
    input_features_dict,
    target_features_dict,
    reverse_target_features_dict,
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
)

st.set_page_config(page_title="Machine Translation", page_icon="🌍", layout="centered")

latent_dim = 256

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "training_model.keras"


@st.cache_resource
def load_translation_model():

    training_model = load_model(MODEL_PATH)

    encoder_inputs = training_model.input[0]

    _, state_h_enc, state_c_enc = training_model.layers[2].output

    encoder_states = [state_h_enc, state_c_enc]

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = training_model.input[1]

    decoder_lstm = training_model.layers[3]

    decoder_dense = training_model.layers[4]

    decoder_state_input_hidden = Input(
        shape=(latent_dim,), name="decoder_state_input_hidden"
    )

    decoder_state_input_cell = Input(
        shape=(latent_dim,), name="decoder_state_input_cell"
    )

    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

    decoder_outputs, state_hidden, state_cell = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )

    decoder_states = [state_hidden, state_cell]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model


encoder_model, decoder_model = load_translation_model()


def sentence_to_matrix(sentence):

    sentence = sentence.strip()

    sentence_matrix = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )

    tokens = re.findall(r"[\w']+|[^\s\w]", sentence)

    for timestep, token in enumerate(tokens):

        if timestep >= max_encoder_seq_length:
            break

        if token in input_features_dict:

            sentence_matrix[0, timestep, input_features_dict[token]] = 1.0

    return sentence_matrix


def clean_translation(text):

    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace("¿ ", "¿")
    text = text.replace("¡ ", "¡")

    while "!!" in text:
        text = text.replace("!!", "!")

    while ".." in text:
        text = text.replace("..", ".")

    return text.strip()


def decode_sequence(test_input):

    states_value = encoder_model.predict(test_input, verbose=0)

    target_seq = np.zeros((1, 1, num_decoder_tokens))

    target_seq[0, 0, target_features_dict["<START>"]] = 1.0

    decoded_sentence = ""

    while True:

        output_tokens, hidden_state, cell_state = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_token = reverse_target_features_dict[sampled_token_index]

        if sampled_token == "<END>":
            break

        decoded_sentence += " " + sampled_token

        if len(decoded_sentence.split()) > max_decoder_seq_length:
            break

        target_seq = np.zeros((1, 1, num_decoder_tokens))

        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [hidden_state, cell_state]

    return clean_translation(decoded_sentence)


st.title("🌍 Machine Translation")
st.subheader("Neural Machine Translation")

st.caption(
    "Seq2Seq Encoder-Decoder LSTM model trained on " "English-Spanish sentence pairs."
)

english_text = st.text_area("Enter English text:")

if st.button("Translate"):

    if english_text.strip():

        matrix = sentence_to_matrix(english_text)

        translation = decode_sequence(matrix)

        st.success(f"Spanish: {translation}")

    else:

        st.warning("Please enter some text.")

with st.expander("Model Statistics"):

    st.write(f"Encoder Vocabulary: {num_encoder_tokens}")

    st.write(f"Decoder Vocabulary: {num_decoder_tokens}")

    st.write(f"Max Input Tokens: {max_encoder_seq_length}")

    st.write(f"Max Output Tokens: {max_decoder_seq_length}")

    st.write(f"Latent Dimension: {latent_dim}")
