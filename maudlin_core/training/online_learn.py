import json 
import yaml
import hmac
import hashlib
import requests
import urllib.parse

import os
import csv
import sys
import signal
import datetime
import random
import numpy as np
import pandas as pd

from pathlib import Path

from ..lib.framework.maudlin import load_maudlin_data
from ..lib.framework.maudlin_unit_config import get_current_unit_config

from ..model.model import create_model, generate_model_file_name
from ..lib.data_loading.training import load_for_training


# BEGIN
maudlin = load_maudlin_data()
config = get_current_unit_config()

def get_data_directory_init():

    data_dir = None

    if config['data']['training_file']:
        data_dir = os.path.dirname(config['data']['training_file'])

        if not os.path.exists(data_dir):
            raise ValueError("The data_file setting in the config is invalid. Path does not exist.")

    else:
        data_dir = maudlin['data-directory'] + "/trainings/" + maudlin['current-unit']
        data_dir += "/run_" + str(len(os.listdir(data_dir)) + 1)

        # Check if the directory already exists
        if os.path.exists(data_dir):
            print('\a')  # Emit a beep
            print(f"Error: No data_file was set, and the directory we created for it, '{data_dir}', already exists.")
            sys.exit(1)  # Exit the program with an error code

        # Create the directory
        os.makedirs(data_dir, exist_ok=True)

    return data_dir

# trainings_data_dir = get_data_directory_init()

MODEL_FILE = generate_model_file_name(maudlin)
DATA_FILE = config['data']['training_file']
TIMESTEPS = config['timesteps']

# Signal handler for saving the model on Ctrl+C
def signal_handler(signal, frame):
    print(f"\nInterrupt received! Saving the model to {MODEL_FILE}...")
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}. Exiting gracefully.")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def read_feedback_file():
    """Read timestep entries from a file."""
    with open(config['feedback_file'], 'r') as file:
        entries = list(map(int, file.read().split()))
    return entries

def append_to_feedback_file(value):
    """Append a single value to the feedback file."""
    with open(config['feedback_file'], 'a') as file:
        file.write(f" {value} ")

def remove_last_feedback_entry():
    """Remove the last feedback entry from the feedback file."""
    with open(config['feedback_file'], 'r+') as file:
        content = file.read().strip()
        if content:
            # Remove the last value from the file
            content = content[:content.rfind(" ")].strip()
            file.seek(0)
            file.write(content)
            file.truncate()


def run_online_learning(trainings_data_dir):
    print("\nStarting the online-learning script...")

    # Get timestamp from first record of data file
    df = pd.read_csv(config['data']['training_file'], nrows=1)

    df_begin_timestamp = df.iloc[0, 0]

    dt = datetime.datetime.fromtimestamp(df_begin_timestamp)
    print("Beginning timestamp: " + dt.strftime("%m/%d/%Y %HH %MM"))

    # Load and preprocess the data
    print("Loading and preprocessing data...")
    generate_targets = False


#    data_loading_function_file_path = get_unit_function_path(maudlin, 'data-loading-function-training')    
 #   data_loading_function = load_function_from_file(data_loading_function_file_path, "apply")

    X, y_unused, feature_count = load_for_training(trainings_data_dir, generate_targets)


    # Load or create the model
    print("Getting the model...")
    model = create_model(TIMESTEPS, feature_count)

    # Dictionary to store feedback for overlapping timesteps
    feedback_memory = {}

    # Feedback entries from file (if provided)
    feedback_from_file = read_feedback_file() 
    feedback_pointer = 0  # Pointer to track usage of file entries

    print(f"Shape of X: {X.shape}")  # Expect [num_samples, timesteps, feature_count]

    print(f"Feedback file: {config['feedback_file']}")

    def train_step(batch, target_labels):
        print("!! TRAIN STEP !!")
        return model.train_on_batch(batch, target_labels)

    # Initialize variables
    undo_stack = []
    batches_stack = []  # To store (batches, feedback)
    batch_index = 0  # Manual index for traversing X

    try:
        print("\n--- BEGIN -----------------------------------")
        while batch_index < len(X):  # Replace `for` loop with a `while` loop
            # Fetch batch at the current index
            batch = X[batch_index].reshape(1, 12, 14).astype("float32")  # Ensure consistent data type

            # Predict the category
            prediction = model.predict(batch, verbose=0)

            # Get feedback for this feature set
            feedback = []
            predictions_stack = []  # Track predictions and confidences for undo

            # Add the current batch-feedback pair to the stack
            if batch_index >= len(batches_stack):
                batches_stack.append((batch, feedback))
            else:
                batches_stack[batch_index] = (batch, feedback)

            timestep_index = 0  # Manual control of timestep_index

            while timestep_index < len(prediction[0]):  # Iterate over timesteps in the current feature set
                timestep_prediction = prediction[0][timestep_index]
                predicted_label = np.argmax(timestep_prediction)
                confidence = timestep_prediction[predicted_label]

                dt = datetime.datetime.fromtimestamp(df_begin_timestamp + ((batch_index * config['timesteps'] * 60) + (timestep_index * 60)))
                strr = dt.strftime("%m/%d/%Y %H:%M")

                correct_label = None

                cache_key = (batch_index, timestep_index)

                # Check cache first
                if cache_key in feedback_memory:
                    correct_label, opl = feedback_memory[cache_key]
                    match_status = f"MATCHES current predicted label {predicted_label} opl={opl}" if predicted_label == correct_label else ""
                    print(f"{batch_index} / {timestep_index + 1}: {strr} P= {predicted_label}, C= {confidence:.2f} > Using feedback from cache {correct_label} {match_status}")                    

                    # Move to the next timestep
                    timestep_index += 1
                else:
                    # Use file entry if available
                    if feedback_pointer < len(feedback_from_file):
                        correct_label = feedback_from_file[feedback_pointer]
                        match_status = "MATCHES" if predicted_label == correct_label else ""
                        print(f"{batch_index} / {timestep_index + 1}: {strr} P= {predicted_label}, C= {confidence:.2f} > Using feedback from file {correct_label} {match_status}")
                        feedback_pointer += 1
                        undo_stack.append((batch_index, timestep_index, correct_label))  # Track entry for undo
                        feedback.append(correct_label)
                        feedback_memory[(batch_index, timestep_index)] = (correct_label, predicted_label)
                        batches_stack[batch_index] = (batch, feedback)  # Update stack

                        timestep_index += 1
                        
                        continue  # Skip user input for file feedback
                    else:
                        # Handle user input when no file feedback is available
                        print()
                        while True:
                            user_input = input(f"{batch_index} / {timestep_index + 1}: {strr} P= {predicted_label}, C= {confidence:.2f} > ").strip().lower()

                            if user_input == 'q':
                                print("\nGracefully exiting and saving the model...")
                                model.save(MODEL_FILE)
                                print("Model saved. Goodbye!")
                                exit()
                            elif user_input == '<':  # Undo
                                if undo_stack:
                                    # Pop the last entry from undo_stack and batches_stack
                                    _, last_timestep_index, _ = undo_stack.pop()
                                    last_batch, last_feedback = batches_stack.pop()

                                    if not last_feedback:  # Handle boundary case
                                        batch_index -= 1  # Rewind batch index
                                        timestep_index = config['timesteps']
                                        last_batch, last_feedback = batches_stack.pop()

                                    # Restore previous feedback and batch 
                                    last_feedback.pop()
                                    batches_stack.append((last_batch, last_feedback))

                                    remove_last_feedback_entry()

                                    # Adjust pointers
                                    feedback_from_file = read_feedback_file()
                                    feedback_pointer = len(feedback_from_file)
                                    del feedback_memory[(batch_index, last_timestep_index)]

                                    timestep_index -= 1  # Adjust for the current loop
                                    break  # Revisit the loop
                                else:
                                    print("Nothing to undo.")
                                    continue
                            elif user_input.isdigit():  # Valid feedback
                                correct_label = int(user_input)

                                if correct_label >= 0 and correct_label < len(config['data_bin_labels']):
                                    undo_stack.append((batch_index, timestep_index, correct_label))
                                    append_to_feedback_file(correct_label)

                                    # Update feedback and feedback_memory
                                    _, lFeedback = batches_stack[-1]
                                    lFeedback.append(correct_label)

                                    feedback_memory[(batch_index, timestep_index)] = (correct_label, predicted_label)

                                    # Move to the next timestep
                                    timestep_index += 1

                                    break

                            else:
                                print(f"\nEnter 0 - {len(config['data_bin_labels'])}, < to undo, q to quit.\n")



            # Check if feedback for this set is complete
            fe, fb = batches_stack[-1]

            if len(fb) == config['timesteps']:
                print(f"Training on batch_index {batch_index} with feedback {fb}")
                target_labels = np.array([fb])  # Feedback as NumPy array
                loss, accuracy = train_step(fe, target_labels)
                print(f"Batch Loss: {loss}, Batch Accuracy: {accuracy}")

                # Move to the next feature set
                batch_index += 1
                timestep_index = 0


    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving the model...")
        model.save(MODEL_FILE)
        print("Model saved. Goodbye!")


    model.save(MODEL_FILE)
    print('\a')
    print("Model saved after completed run! One down!")

    xx = input("\nThis batch has completed feedback! Do you want to start with a fresh batch next time? ").lower().startswith('y')

    if xx:
        config['data']['training_file'] = None
        config['feedback_file'] = None

        with open(maudlin['data-directory'] + "/configs/" + maudlin['current-unit'] + ".config.yaml", "w") as file:
            yaml.dump(config, file, indent=4)

        print("\nOkay. We'll start anew next time.")
    else:
        print("\nOkay. Nothing changes then.")



def call_the_api(get_random_immediate_past_data=False):
    api_key = "mx0vglxQ3stSrDNttt"
    secret_key = "6de54e87053b4193ac72cb5942faa9ac"
    base_url = "https://api.mexc.com/api/v3/klines"
    
    # Define query parameters
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
    }

    if get_random_immediate_past_data:
        # Get the current date and time
        now = datetime.datetime.now()

        # Subtract a random number of days (1 to 7) from the current date
        random_days_ago = random.randint(1, 7)
        apastday = now - datetime.timedelta(days=random_days_ago)

        # Set the time to 12:01 AM
        apastday = apastday.replace(hour=0, minute=1, second=0, microsecond=0)

        # Convert to a timestamp in milliseconds
        timestamp = int(apastday.timestamp() * 1000)

        # Set the params for the start and end times
        params['startTime'] = str(timestamp)
        params['endTime'] = str(timestamp + (1000 * 60 * 499))  # 499 minutes later

        print(f"Using Random Immediate Past Data {params}")
    else:
        pass   # will retrieve the last 500 minutes

    # Create the query string
    query = "&".join(f"{key}={urllib.parse.quote_plus(str(value))}" for key, value in params.items())

    # Generate HMAC SHA256 signature
    signature = hmac.new(
        secret_key.encode("utf-8"), 
        query.encode("utf-8"), 
        hashlib.sha256
    ).hexdigest()

    # Add the signature to the query parameters
    query += f"&signature={urllib.parse.quote_plus(signature)}"

    # Final URL with query string
    url = f"{base_url}?{query}"

    # Define headers
    headers = {
        "X-MEXC-APIKEY": api_key,
        "Accept": "application/json"
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse and return JSON response
        return response.json()

    # Handle non-200 responses
    print(f"Error: {response.status_code}, {response.text}")
    return None

def save_api_response_in_trainings_folder(trainings_data_dir, data):
    # Extract the timestamp from the first element of the data
    if not data or not data[0]:
        raise ValueError("Data is empty or invalid format.")
    
    base_timestamp = str(data[0][0])  # Use the first timestamp from the data
    
    # Create the base filename
    base_filename = f"mexc-{base_timestamp}.json"
    filename = base_filename
    filepath = os.path.join(trainings_data_dir, filename)
    
    # Check for existing files and append suffix if necessary
    suffix = 'A'
    while os.path.exists(filepath):
        filename = f"mexc-{base_timestamp}-{suffix}.json"
        filepath = os.path.join(trainings_data_dir, filename)
        suffix = chr(ord(suffix) + 1)  # Increment suffix alphabetically (A, B, C, ...)

    # Save response to JSON file
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)

    return filepath

def convert_json_to_csv(json_file, csv_file):
    # Load the JSON data from the file
    with open(json_file, "r") as file:
        data = json.load(file)

    # Prepare the CSV header
    csv_header = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]

    # Open the CSV file for writing
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(csv_header)

        # Write the rows of data
        for row in data:
            # Extract relevant fields and normalize timestamp to seconds
            timestamp = row[0] / 1000  # Convert milliseconds to seconds
            open_price = row[1]
            high = row[2]
            low = row[3]
            close = row[4]
            volume = row[5]

            # Write the row to the CSV file
            writer.writerow([timestamp, open_price, high, low, close, volume])

    print(f"CSV file '{csv_file}' created successfully.")

    return csv_file


if __name__ == "__main__":

    data_dir = None

    if config['use_online_learning'] == True:

        data_dir = get_data_directory_init()

        if not config["data_file"]:
            xx = input("You don't have a data file set. Do you want to download new fresh data? ").lower().startswith('y')

            if xx:
                # call the API,  get the data as json, 
                randomize_date_range_of_data = True
                api_data = call_the_api(randomize_date_range_of_data)

                #save in trainings folder, 
                filename = save_api_response_in_trainings_folder(data_dir, api_data)

                #convert to csv, save that in trainings folder
                csv_filename = convert_json_to_csv(filename, filename.replace(".json", ".csv"))
                config['data']['training_file'] = csv_filename

                feedback_filename = filename.replace(".json", ".feedback.txt")
                config['feedback_file'] = feedback_filename

                cyp = maudlin['data-directory'] + "/configs/" + maudlin['current-unit'] + ".config.yaml"
                with open(cyp, "w") as file:
                    yaml.dump(config, file, indent=4)

                # touch feedback file
                Path(feedback_filename).touch()
            else:
                print("Exiting. You need to set a data file.")
                exit()

        elif not config['feedback_file']:
            filename = config['data']['training_file']
            feedback_filename = filename.replace(".csv", ".feedback.txt")
            config['feedback_file'] = feedback_filename

            Path(feedback_filename).touch()

            with open(maudlin['data-directory'] + "/configs/" + maudlin['current-unit'] + ".config.yaml", "w") as file:
                yaml.dump(config, file, indent=4)

        run_online_learning(data_dir)

    else:
        print(f"Current Unit [{maudlin['current-unit']}] does not have the use_online_learning config set to True.")
        print()
