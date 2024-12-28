#!/bin/bash

# Maudlin Main Script: mdln
#
# # Determine the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script's directory
cd "$SCRIPT_DIR" || exit

# Configurations
DEFAULT_DATA_DIR="$HOME/src/_data/maudlin"
DEFAULT_CONFIG_FILE="default.config.yaml"
DATA_YAML="$DEFAULT_DATA_DIR/maudlin.data.yaml"

CURRENT_UNIT=$(yq ".current-unit" $DATA_YAML)

# Functions
initialize_maudlin() {
    echo "Initializing Maudlin directory structure..."
    mkdir -p "$DEFAULT_DATA_DIR/configs" \
             "$DEFAULT_DATA_DIR/models" \
             "$DEFAULT_DATA_DIR/functions" \
             "$DEFAULT_DATA_DIR/inputs" \
             "$DEFAULT_DATA_DIR/predictions"
             
    # Git init for configs
    if [ ! -d "$DEFAULT_DATA_DIR/configs/.git" ]; then
        echo "Setting up git for configs..."
        git -C "$DEFAULT_DATA_DIR/configs" init
        git -C "$DEFAULT_DATA_DIR/functions" init
    fi

    # Create maudlin.data.yaml if it doesn't exist
    if [ ! -f "$DATA_YAML" ]; then
        echo "Creating maudlin.data.yaml..."
        echo "units: []" > "$DATA_YAML"
        echo "data-directory: $DEFAULT_DATA_DIR" >> "$DATA_YAML"
    fi

    echo "Maudlin initialization complete."
}

list_units() {
    echo "Listing all managed units:"
    if [ -f "$DATA_YAML" ]; then
        yq '.units | keys' "$DATA_YAML"  # Requires yq for YAML parsing
    else
        echo "Maudlin data file not found. Perhaps you need to initialize Maudlin first using 'mdln init'."
    fi
}

set_current_unit_name() {
    UNIT_NAME="$1"

    # Check if the unit exists
    if ! verify_unit_exists "$UNIT_NAME"; then
        echo "Error: Unit name '$UNIT_NAME' does not exist. Use 'mdln list' to see available units."
        exit 1
    fi

    # Update the current unit in maudlin.data.yaml
    yq -i ".current-unit = \"$UNIT_NAME\"" "$DATA_YAML"
    echo "Current unit set to '$UNIT_NAME'."
}

set_current_run() {
    RUN_ID="$1"
    RUN_NAME="run_${RUN_ID##run_}"
    CURRENT_UNIT=$(yq ".current-unit" "$DATA_YAML")

    RUN_CONFIG_DIR="$DEFAULT_DATA_DIR/trainings/$CURRENT_UNIT/$RUN_NAME"
    DEFAULT_CONFIG="$DEFAULT_DATA_DIR/configs/$CURRENT_UNIT.config.yaml"

    # Check if the run config exists
    if [ -f "$RUN_CONFIG_DIR/config.yaml" ]; then
        cp "$RUN_CONFIG_DIR/config.yaml" "$DEFAULT_CONFIG"
        echo "Run configuration '$RUN_NAME' set as default config."
    else
        echo "Error: Configuration for '$RUN_NAME' does not exist."
        exit 1
    fi
}

is_run_number() {
    INPUT="$1"
    if [[ "$INPUT" =~ ^run_[0-9]+$ || "$INPUT" =~ ^[0-9]+$ ]]; then
        return 0 # True
    else
        return 1 # False
    fi
}

set_current_unit() {
    verify_maudlin_data_file_exists

    UNIT_NAME="$1"

    # Ensure UNIT_NAME is provided
    if [ -z "$UNIT_NAME" ]; then
        echo "Usage: mdln use <name|run_number>"
        exit 1
    fi

    if is_run_number "$UNIT_NAME"; then
        set_current_run "$UNIT_NAME"
    else
        set_current_unit_name "$UNIT_NAME"
    fi
}

new_unit() {
    verify_maudlin_data_file_exists

    UNIT_NAME=""
    TRAINING_CSV_PATH=""
    PREDICTION_CSV_PATH=""

    # Parse named parameters
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --training-csv-path=*)
                TRAINING_CSV_PATH="${1#*=}"
                ;;
            --prediction-csv-path=*)
                PREDICTION_CSV_PATH="${1#*=}"
                ;;
            *)
                if [ -z "$UNIT_NAME" ]; then
                    UNIT_NAME="$1"
                else
                    echo "Error: Unexpected parameter '$1'"
                    echo "Usage: mdln new <name> --training-csv-path=<path> --prediction-csv-path=<path>"
                    exit 1
                fi
                ;;
        esac
        shift
    done

    # Ensure UNIT_NAME is provided
    if [ -z "$UNIT_NAME" ]; then
        echo "Error: Unit name is required."
        echo "Usage: mdln new <name> --training-csv-path=<path> --prediction-csv-path=<path>"
        exit 1
    fi

    # Ensure paths are provided
    if [ -z "$TRAINING_CSV_PATH" ]; then
        echo "Error: --training-csv-path is required."
        exit 1
    fi

    if [ -z "$PREDICTION_CSV_PATH" ]; then
        echo "Error: --prediction-csv-path is required."
        exit 1
    fi

    # Ensure unique unit name
    if verify_unit_exists "$UNIT_NAME"; then 
        echo "Error: Unit name '$UNIT_NAME' already exists. Please choose a unique name."
        exit 1
    fi

    # Prepare new unit
    echo "Creating a new unit '$UNIT_NAME'..."
    CONFIG_SLUG="/configs/$UNIT_NAME.config.yaml"
    CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_SLUG"

    INPUT_FUNCTION_SLUG="/functions/$UNIT_NAME.input_function.py"
    INPUT_FUNCTION_PATH="$DEFAULT_DATA_DIR/$INPUT_FUNCTION_SLUG"

    touch $INPUT_FUNCTION_PATH
    echo "# Blank input function for $UNIT_NAME" > "$INPUT_FUNCTION_PATH"

    TARGET_FUNCTION_SLUG="/functions/$UNIT_NAME.target_function.py"
    TARGET_FUNCTION_PATH="$DEFAULT_DATA_DIR/$TARGET_FUNCTION_SLUG"

    touch $TARGET_FUNCTION_PATH
    echo "# Blank target function for $UNIT_NAME" > "$TARGET_FUNCTION_PATH"

    PRE_PREDICTION_FUNCTION_SLUG="/functions/$UNIT_NAME.pre_prediction_function.py"
    PRE_PREDICTION_FUNCTION_PATH="$DEFAULT_DATA_DIR$PRE_PREDICTION_FUNCTION_SLUG"

    touch $PRE_PREDICTION_FUNCTION_PATH
    echo "\ndef apply(folder, predictions):\n\tpass\n" > "$PRE_PREDICTION_FUNCTION_PATH"

    PRE_TRAINING_FUNCTION_SLUG="/functions/$UNIT_NAME.pre_training_function.py"
    PRE_TRAINING_FUNCTION_PATH="$DEFAULT_DATA_DIR$PRE_TRAINING_FUNCTION_SLUG"

    touch $PRE_TRAINING_FUNCTION_PATH
    echo "\ndef apply(model, X_train, y_train, X_test, y_test):\n\tpass\n" > "$PRE_TRAINING_FUNCTION_PATH"

    POST_PREDICTION_FUNCTION_SLUG="/functions/$UNIT_NAME.post_prediction_function.py"
    POST_PREDICTION_FUNCTION_PATH="$DEFAULT_DATA_DIR$POST_PREDICTION_FUNCTION_SLUG"

    touch $POST_PREDICTION_FUNCTION_PATH
    echo "\ndef apply(folder, predictions):\n\tpass\n" > "$POST_PREDICTION_FUNCTION_PATH"

    POST_TRAINING_FUNCTION_SLUG="/functions/$UNIT_NAME.post_training_function.py"
    POST_TRAINING_FUNCTION_PATH="$DEFAULT_DATA_DIR$POST_TRAINING_FUNCTION_SLUG"

    touch $POST_TRAINING_FUNCTION_PATH
    echo "\ndef apply(model, X_train, y_train, X_test, y_test):\n\tpass\n" > "$POST_TRAINING_FUNCTION_PATH"

    # Copy the default config file to the new config
    cp "./$DEFAULT_CONFIG_FILE" "$CONFIG_PATH"
    sed -i -E "s|^\s*training_file:.*|& $TRAINING_CSV_PATH|" "$CONFIG_PATH"
    sed -i -E "s|^\s*prediction_file:.*|& $PREDICTION_CSV_PATH|" "$CONFIG_PATH"

    # Commit the config file
    pushd "$DEFAULT_DATA_DIR/configs" > /dev/null
    git add "$UNIT_NAME.config.yaml"
    git commit -m "Add config for $UNIT_NAME"
    CONFIG_COMMIT_ID=$(git rev-parse HEAD)
    popd > /dev/null

    # Add to maudlin.data.yaml
    yq -i ".units[\"$UNIT_NAME\"] = {\"config-commit-id\": \"$CONFIG_COMMIT_ID\", \"config-path\": \"$CONFIG_SLUG\", \"keras-filename\": null, \"data-filename\": \"/inputs/${UNIT_NAME}-data\", \"input-function\": \"$INPUT_FUNCTION_SLUG\", \"target-function\": \"$TARGET_FUNCTION_SLUG\", \"pre-training-function\": \"$PRE_TRAINING_FUNCTION_SLUG\", \"pre-prediction-function\": \"$PRE_PREDICTION_FUNCTION_SLUG\", \"post-training-function\": \"$POST_TRAINING_FUNCTION_SLUG\", \"post-prediction-function\": \"$POST_PREDICTION_FUNCTION_SLUG\"}" "$DATA_YAML"

    echo "Unit '$UNIT_NAME' created successfully with data file: $TRAINING_CSV_PATH"
}

show_current_unit() {
    verify_maudlin_data_file_exists
    verify_current_unit_is_set

    # Retrieve and display properties of the current unit
    if ! verify_unit_exists $CURRENT_UNIT; then
        echo "Error: Current unit '$CURRENT_UNIT' not found in units list. Check your maudlin.data.yaml file."
        exit 1
    fi

    DATA_DIR=$(yq '.data-directory' "$DATA_YAML")

    echo ""
    echo "Current Unit: $CURRENT_UNIT"
    echo "Data directory: $DATA_DIR"
    echo ""
    echo "Properties:"
    yq ".units.${CURRENT_UNIT} | to_entries[] | \"\(.key): \(.value)\"" "$DATA_YAML"

    CONFIG_PATH=$(yq ".units.${CURRENT_UNIT}.config-path" $DATA_YAML)
    FULL_CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_PATH"

    USE_ONLINE_LEARNING_MODE=$(yq ".use_online_learning" $FULL_CONFIG_PATH)
    if [ "$USE_ONLINE_LEARNING_MODE" = "true" ] || [ "$USE_ONLINE_LEARNING_MODE" = "True" ]; then
      echo ""
      echo "ONLINE LEARN mode"
      echo "Data file (training): $(yq '.data_file_training' $FULL_CONFIG_PATH)"
      echo "Data file (prediction): $(yq '.data_file_prediction' $FULL_CONFIG_PATH)"
      echo "Feedback file: $(yq '.feedback_file' $FULL_CONFIG_PATH)"
    else
      echo ""
      echo "BATCH LEARNING mode"
    fi

    echo ""
}

edit_current_unit() {
    verify_maudlin_data_file_exists
    verify_current_unit_is_set
    
    # Retrieve config and target-function file paths for the current unit
    CONFIG_PATH=$(yq ".units.${CURRENT_UNIT}.config-path" $DATA_YAML)
    TARGET_FUNCTION_SLUG=$(yq ".units.${CURRENT_UNIT}.target-function" $DATA_YAML)

    if [ -z "$CONFIG_PATH" ] || [ -z "$TARGET_FUNCTION_SLUG" ]; then
        echo "Error: Config or target-function paths not found for the current unit '$CURRENT_UNIT'."
        exit 1
    fi

    FULL_CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_PATH"
    FULL_TARGET_FUNCTION_PATH="$DEFAULT_DATA_DIR$TARGET_FUNCTION_SLUG"

    # Check if the files exist before opening
    if [ ! -f "$FULL_CONFIG_PATH" ]; then
        echo "Error: Config file '$FULL_CONFIG_PATH' does not exist."
        exit 1
    fi

    if [ ! -f "$FULL_TARGET_FUNCTION_PATH" ]; then
        echo "Error: Target function file '$FULL_TARGET_FUNCTION_PATH' does not exist."
        exit 1
    fi

    echo "Opening config and function files for unit '$CURRENT_UNIT' in lvim..."
    lvim "$FULL_CONFIG_PATH" "$DEFAULT_DATA_DIR/functions/$CURRENT_UNIT"*.py -p
}

verify_maudlin_data_file_exists() {
  if [ ! -f "$DATA_YAML" ]; then
    echo "Maudlin data file not found. Perhaps you need to initialize Maudlin first using 'mdln init'."
    exit 1
  fi
}

verify_current_unit_is_set() {
  # Check if the current unit is set
  CU=$(yq '.current-unit' "$DATA_YAML")
  if [ "$CU" == "null" ]; then
    echo "No current unit is set. Use 'mdln use <name>' to set a current unit."
    exit 1
  fi
}

verify_unit_exists() {
    local unit_name=$1
    # Check if the unit exists in the YAML file
    CU=$(yq ".units.${unit_name}" "$DATA_YAML")
    if [ ! "$CU" == "null" ]; then
        return 0  # Success: Unit exists
    else
        return 1  # Failure: Unit does not exist
    fi
}

clean_unit_output() {
  # remove the model for the current unit
  MODEL_SLUG=$(yq ".units.${CURRENT_UNIT}.keras-filename" $DATA_YAML)
  DATA_DIR=$(yq ".data-directory" $DATA_YAML)

  echo "Giving you 7 seconds to come to your senses..."
  sleep 3

  echo "..3 seconds more.."
  sleep 3

  echo "..okay, here we go!"
  sleep 1

  rm $DATA_DIR$MODEL_SLUG

  yq -i ".units.\"${CURRENT_UNIT}\".keras-filename = null" "$DATA_YAML"

  echo "Done. Removed model for ${CURRENT_UNIT}"
}

run_predictions() {
  python3 /home/jjames/src/learning/btcmodel/btcmodel/predicting/predict.py
}


run_training() {
  # Default values
  EPOCHS=""
  TRAINING_RUN_ID=""

  # Parse named parameters
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      -e|--epochs)
        EPOCHS=$2
        shift 2
        ;;
      -r|--run-id)
        TRAINING_RUN_ID=$2
        shift 2
        ;;
      *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
  done

  # Retrieve config path for the current unit
  CONFIG_PATH=$(yq ".units.${CURRENT_UNIT}.config-path" "$DATA_YAML")
  if [ -z "$CONFIG_PATH" ]; then
    echo "Error: Config path not found for the current unit '$CURRENT_UNIT'."
    exit 1
  fi

  FULL_CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_PATH"

  # Ensure the config file exists before proceeding
  if [ ! -f "$FULL_CONFIG_PATH" ]; then
    echo "Error: Config file '$FULL_CONFIG_PATH' does not exist."
    exit 1
  fi

  # Update epochs in the config if specified
  if [ -n "$EPOCHS" ]; then
    yq -i ".epochs = $EPOCHS" "$FULL_CONFIG_PATH"
    echo "Updated epochs to $EPOCHS in $FULL_CONFIG_PATH"
  fi

  # Retrieve the value of USE_ONLINE_LEARNING_MODE from the YAML config
  USE_ONLINE_LEARNING_MODE=$(yq ".use_online_learning" "$FULL_CONFIG_PATH")

  # Handle training run ID
  if [ -n "$TRAINING_RUN_ID" ]; then
    TRAINING_RUN_PATH="$DEFAULT_DATA_DIR/trainings/$CURRENT_UNIT/run_$TRAINING_RUN_ID"

    # Ensure the training run directory exists
    if [ ! -d "$TRAINING_RUN_PATH" ]; then
      echo "Error: Training run directory '$TRAINING_RUN_PATH' does not exist."
      exit 1
    fi

    # Execute the training script based on mode
    if [ "$USE_ONLINE_LEARNING_MODE" = "true" ] || [ "$USE_ONLINE_LEARNING_MODE" = "True" ]; then
      cd /home/jjames/src/learning/btcmodel/ || exit
      python3 -m btcmodel.training.online_learn.online_learn "$TRAINING_RUN_PATH"
    else
      cd /home/jjames/src/learning/btcmodel/ || exit
      python3 -m btcmodel.training.batch.batch "$TRAINING_RUN_PATH"
    fi
  else
    # No training run ID provided, call the script without parameters
    if [ "$USE_ONLINE_LEARNING_MODE" = "true" ] || [ "$USE_ONLINE_LEARNING_MODE" = "True" ]; then
      cd /home/jjames/src/learning/btcmodel/ || exit
      python3 -m btcmodel.training.online_learn.online_learn
    else
      cd /home/jjames/src/learning/btcmodel/ || exit
      python3 -m btcmodel.training.batch.batch
    fi
  fi
}

cd_to_data_dir() {
    verify_maudlin_data_file_exists
    verify_current_unit_is_set

    # Retrieve and display properties of the current unit
    if ! verify_unit_exists $CURRENT_UNIT; then
        echo "Error: Current unit '$CURRENT_UNIT' not found in units list. Check your maudlin.data.yaml file."
        exit 1
    fi

    DATA_DIR=$(yq '.data-directory' "$DATA_YAML")

    echo $DATA_DIR
    cd $DATA_DIR
}

compare_runs() {
  if [[ -z "$1" || -z "$2" ]]; then
    echo "Usage: mdln compare <RUN1> <RUN2>"
    exit 1
  fi
  RUN1=$1
  RUN2=$2

  # Load the data directory and current unit
  DATA_DIR=$(yq ".data-directory" "$DATA_YAML")
  
  # Resolve paths for the two runs
  RUN_PATH1="$DATA_DIR/trainings/$CURRENT_UNIT/run_$RUN1"
  RUN_PATH2="$DATA_DIR/trainings/$CURRENT_UNIT/run_$RUN2"
  
  # Check if both run paths exist
  if [[ ! -d "$RUN_PATH1" || ! -d "$RUN_PATH2" ]]; then
    echo "Error: One or both run directories do not exist:"
    echo "  $RUN_PATH1"
    echo "  $RUN_PATH2"
    exit 1
  fi

  # Navigate to the project directory where the Python module is located
  cd /home/jjames/src/learning/btcmodel/ || exit

  # Run the Python compare_runs script
  python3 -m btcmodel.training.compare_runs.compare_runs "$RUN_PATH1" "$RUN_PATH2"

  # Check the result of the Python script
  if [[ $? -ne 0 ]]; then
    echo "Error: Python script for compare_runs failed."
    exit 1
  fi

  # Open the visual diffs directory automatically after comparison
  echo feh "$DATA_DIR/trainings/$CURRENT_UNIT/run_$RUN1/comparison/vs_run_$RUN2"

}

history() {
  # Optional flag for timeline view
  TIMELINE=false
  if [[ "$1" == "--timeline" ]]; then
    TIMELINE=$1
  fi

  # Load unit info
  DATA_DIR=$(yq ".data-directory" "$DATA_YAML")
  HISTORY_FILE="$DATA_DIR/trainings/$CURRENT_UNIT/history.yaml"

  # Check if history.yaml exists
  if [[ ! -f "$HISTORY_FILE" ]]; then
    echo "Error: No history file found for unit '$CURRENT_UNIT'."
    exit 1
  fi

  # Run Python script for history visualization
  cd /home/jjames/src/learning/btcmodel/ || exit
  python3 -m btcmodel.exploring.visualize_history "$TIMELINE"

  if [[ $? -ne 0 ]]; then
    echo "Error: Python script for history visualization failed."
    exit 1
  fi
}

navigate() {
  # Load the history file
  DATA_DIR=$(yq ".data-directory" "$DATA_YAML")
  HISTORY_FILE="$DATA_DIR/trainings/$CURRENT_UNIT/history.yaml"

  # Check if history.yaml exists
  if [[ ! -f "$HISTORY_FILE" ]]; then
    echo "Error: No history file found for unit '$CURRENT_UNIT'."
    exit 1
  fi

  # Start interactive navigation
  cd /home/jjames/src/learning/btcmodel/ || exit
  python3 -m btcmodel.history.navigate_runs "$HISTORY_FILE"

  if [[ $? -ne 0 ]]; then
    echo "Error: Python script for navigating runs failed."
    exit 1
  fi
}

msg() {
  if [[ -z "$1" ]]; then
    echo "Usage: mdln msg \"Your message here\""
    exit 1
  fi

  MESSAGE=$1

  # Load the current run ID
  DATA_DIR=$(yq ".data-directory" "$DATA_YAML")
  HISTORY_FILE="$DATA_DIR/trainings/$CURRENT_UNIT/history.yaml"

  # Run Python script to add message
  cd /home/jjames/src/learning/btcmodel/ || exit
  python3 -m btcmodel.history.add_message "$HISTORY_FILE" "$MESSAGE"

  if [[ $? -ne 0 ]]; then
    echo "Error: Python script for adding message failed."
    exit 1
  fi
}


# Main
COMMAND="$1"
shift

case "$COMMAND" in
    init)
        initialize_maudlin
        ;;
    list)
        list_units
        ;;
    new)
        new_unit "$@"
        ;;
    use)
        set_current_unit "$@"
        ;;
    show)
        show_current_unit
        ;;
    edit)
        edit_current_unit
        ;;
    clean)
        clean_unit_output
        ;;
    predict)
        run_predictions
        ;;
    train)
        run_training "$@"
        ;;
    compare)
        compare_runs "$@"
        ;;
    dir)
        cd_to_data_dir
        ;;
    history)
        history "$@"
        ;;
    navigate)
        navigate
        ;;
    msg)
        msg "$@"
        ;;
    *)
        echo "Usage: mdln {init | list | new | use | show | edit | clean | predict | train | dir}"
        exit 1
        ;;
esac

