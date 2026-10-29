from pathlib import Path
import numpy as np

from keras.layers import Input
from keras.models import Model, load_model

from preprocessing import (
    input_docs,
    encoder_input_data,
    input_features_dict,
    target_features_dict,
    reverse_target_features_dict,
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens
)

latent_dim = 256

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "training_model.keras"

print("\nLoading model...\n")

training_model = load_model(MODEL_PATH)

encoder_inputs = training_model.input[0]

encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output

encoder_states = [state_h_enc, state_c_enc]

encoder_model = Model(
    encoder_inputs,
    encoder_states
)

decoder_inputs = training_model.input[1]

decoder_lstm = training_model.layers[3]

decoder_dense = training_model.layers[4]

decoder_state_input_hidden = Input(
    shape=(latent_dim,),
    name="decoder_state_input_hidden"
)

decoder_state_input_cell = Input(
    shape=(latent_dim,),
    name="decoder_state_input_cell"
)

decoder_states_inputs = [
    decoder_state_input_hidden,
    decoder_state_input_cell
]

decoder_outputs, state_hidden, state_cell = decoder_lstm(
    decoder_inputs,
    initial_state=decoder_states_inputs
)

decoder_states = [
    state_hidden,
    state_cell
]

decoder_outputs = decoder_dense(
    decoder_outputs
)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)


def decode_sequence(test_input):

    states_value = encoder_model.predict(
        test_input,
        verbose=0
    )

    target_seq = np.zeros(
        (1, 1, num_decoder_tokens)
    )

    target_seq[0, 0, target_features_dict["<START>"]] = 1.0

    decoded_sentence = ""

    stop_condition = False

    while not stop_condition:

        output_tokens, hidden_state, cell_state = decoder_model.predict(
            [target_seq] + states_value,
            verbose=0
        )

        sampled_token_index = np.argmax(
            output_tokens[0, -1, :]
        )

        sampled_token = reverse_target_features_dict[
            sampled_token_index
        ]

        if sampled_token == "<END>":
            break

        decoded_sentence += " " + sampled_token

        if len(decoded_sentence.split()) > max_decoder_seq_length:
            break

        target_seq = np.zeros(
            (1, 1, num_decoder_tokens)
        )

        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [
            hidden_state,
            cell_state
        ]

    return decoded_sentence.strip()


def clean_translation(text):

    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace("¿ ", "¿")
    text = text.replace("¡ ", "¡")

    words = text.split()

    cleaned_words = []

    for word in words:

        if cleaned_words and word == cleaned_words[-1]:
            continue

        cleaned_words.append(word)

    text = " ".join(cleaned_words)

    while "!!" in text:
        text = text.replace("!!", "!")

    while ".." in text:
        text = text.replace("..", ".")

    while "??" in text:
        text = text.replace("??", "?")

    return text.strip()


def sentence_to_matrix(sentence):

    import re

    sentence = sentence.strip()

    sentence_matrix = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype="float32"
    )

    tokens = re.findall(
        r"[\w']+|[^\s\w]",
        sentence
    )

    print("\nDetected tokens:")
    print(tokens)

    unknown_tokens = []

    for timestep, token in enumerate(tokens):

        if timestep >= max_encoder_seq_length:
            break

        if token in input_features_dict:

            sentence_matrix[
                0,
                timestep,
                input_features_dict[token]
            ] = 1.

        else:

            unknown_tokens.append(token)

    if unknown_tokens:

        print("\nUnknown vocabulary:")
        for token in unknown_tokens:
            print(f" - {token}")

    return sentence_matrix


def show_model_statistics():

    model_size_mb = (
        MODEL_PATH.stat().st_size
        / 1024
        / 1024
    )

    print("\n")
    print("=" * 50)
    print("MODEL STATISTICS")
    print("=" * 50)

    print(f"Encoder vocabulary : {num_encoder_tokens}")
    print(f"Decoder vocabulary : {num_decoder_tokens}")

    print(f"Maximum input tokens : {max_encoder_seq_length}")
    print(f"Maximum output tokens: {max_decoder_seq_length}")

    print(f"Latent dimension : {latent_dim}")

    print(
        f"Model size (MB): "
        f"{model_size_mb:.2f}"
    )

    print(
        f"\nModel exists: "
        f"{MODEL_PATH.exists()}"
    )

    print(
        f"Model path:\n{MODEL_PATH}"
    )

    input(
        "\nPress Enter to return..."
    )

    print()


def test_dataset_examples():

    print("\nDataset Translation Examples\n")

    for seq_index in range(20):

        test_input = encoder_input_data[
            seq_index: seq_index + 1
        ]

        decoded_sentence = clean_translation(
            decode_sequence(test_input)
        )

        print("-")
        print("Input sentence :", input_docs[seq_index])
        print("Translation    :", decoded_sentence)


def custom_translation():

    print("\n" + "=" * 50)
    print("CUSTOM TRANSLATOR")
    print("=" * 50)
    print("Type 'quit' to return to the menu.\n")

    while True:

        try:

            sentence = input("English: ").strip()

            if sentence.lower() == "quit":
                print()
                return

            test_input = sentence_to_matrix(
                sentence
            )

            translated_sentence = clean_translation(
                decode_sequence(test_input)
            )

            print(f"Spanish: {translated_sentence}\n")

        except KeyboardInterrupt:

            print("\n\nReturning to menu...\n")
            return


while True:

    print("\n")
    print("=" * 50)
    print(" MACHINE TRANSLATION SYSTEM ")
    print("=" * 50)

    print("1 - Translate custom text")
    print("2 - Test dataset examples")
    print("3 - Show model statistics")
    print("4 - Exit")

    choice = input("\nChoose option: ").strip()

    if choice == "1":

        custom_translation()

    elif choice == "2":

        test_dataset_examples()

    elif choice == "3":

        show_model_statistics()

    elif choice == "4":

        print("\nGoodbye!\n")
        break

    else:

        print("\nInvalid option. Please try again.\n")