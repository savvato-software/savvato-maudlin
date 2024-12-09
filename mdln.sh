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
        echo "most-recently-used-data-file: null" >> "$DATA_YAML"
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

set_current_unit() {
    verify_data_file_exists

    UNIT_NAME="$1"

    # Ensure UNIT_NAME is provided
    if [ -z "$UNIT_NAME" ]; then
        echo "Usage: mdln use <name>"
        exit 1
    fi

    # Check if the unit exists
    if ! verify_unit_exists "$UNIT_NAME"; then
        echo "Error: Unit name '$UNIT_NAME' does not exist. Use 'mdln list' to see available units."
        exit 1
    fi

    # Update the current unit in maudlin.data.yaml
    yq -i ".current-unit = \"$UNIT_NAME\"" "$DATA_YAML"
    echo "Current unit set to '$UNIT_NAME'."
}


new_unit() {
    verify_data_file_exists

    UNIT_NAME="$1"
    MODEL_TESTING_DATA_PATH="$2"

    # Ensure UNIT_NAME is provided
    if [ -z "$UNIT_NAME" ]; then
        echo "Usage: mdln new <name> [<data-path>]"
        exit 1
    fi

    # If MODEL_TESTING_DATA_PATH is not supplied, use the most-recently-used-data-file
    if [ -z "$MODEL_TESTING_DATA_PATH" ]; then
        MODEL_TESTING_DATA_PATH=$(yq '.most-recently-used-data-file' "$DATA_YAML")
        if [ "$MODEL_TESTING_DATA_PATH" == "null" ] || [[ ! "$MODEL_TESTING_DATA_PATH" =~ ^/ ]]; then
            echo "Error: No model input data-path supplied, and no valid most-recently-used-data-file is set in $DATA_YAML."
            exit 1
        fi
    fi

    # Ensure unique unit name
    if verify_unit_exists "$UNIT_NAME"; then 
        echo "Unit name '$UNIT_NAME' already exists. Please choose a unique name."
        exit 1
    fi

    ln -s $MODEL_TESTING_DATA_PATH $DEFAULT_DATA_DIR/inputs/$UNIT_NAME-data

    # Prepare new unit
    echo "Creating a new unit '$UNIT_NAME'..."
    CONFIG_SLUG="/configs/$UNIT_NAME.config.yaml"
    CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_SLUG"

    TARGET_FUNCTION_SLUG="/functions/$UNIT_NAME.target_function.py"
    TARGET_FUNCTION_PATH="$DEFAULT_DATA_DIR/$TARGET_FUNCTION_SLUG"

    touch $TARGET_FUNCTION_PATH
    echo "# Blank target function for $UNIT_NAME" > "$TARGET_FUNCTION_PATH"

    PREDICTION_DATA_LOADING_FUNCTION_SLUG="/functions/$UNIT_NAME.prediction.data_loading_function.py"
    PREDICTION_DATA_LOADING_FUNCTION_PATH="$DEFAULT_DATA_DIR/$PREDICTION_DATA_LOADING_FUNCTION_SLUG"

    touch $PREDICTION_DATA_LOADING_FUNCTION_PATH
    echo "# Blank prediction data loading function for $UNIT_NAME" > "$PREDICTION_DATA_LOADING_FUNCTION_PATH"

    TRAINING_DATA_LOADING_FUNCTION_SLUG="/functions/$UNIT_NAME.training.data_loading_function.py"
    TRAINING_DATA_LOADING_FUNCTION_PATH="$DEFAULT_DATA_DIR/$TRAINING_DATA_LOADING_FUNCTION_SLUG"

    touch $TRAINING_DATA_LOADING_FUNCTION_PATH
    echo "# Blank training data loading function for $UNIT_NAME" > "$TRAINING_DATA_LOADING_FUNCTION_PATH"

    # Copy the default config file to the new config
    cp "./$DEFAULT_CONFIG_FILE" "$CONFIG_PATH"
    sed -i "s|^data_file:.*|data_file: $DATA_PATH|" "$CONFIG_PATH"

    # Commit the config file
    pushd "$DEFAULT_DATA_DIR/configs" > /dev/null
    git add "$UNIT_NAME.config.yaml"
    git commit -m "Add config for $UNIT_NAME"
    CONFIG_COMMIT_ID=$(git rev-parse HEAD)
    popd > /dev/null

    # Add to maudlin.data.yaml
    yq -i ".units[\"$UNIT_NAME\"] = {\"config-commit-id\": \"$CONFIG_COMMIT_ID\", \"config-path\": \"$CONFIG_SLUG\", \"keras-filename\": null, \"data-filename\": \"/inputs/${UNIT_NAME}-data\", \"data-loading-function-training\": \"$TRAINING_DATA_LOADING_FUNCTION_PATH\", \"data-loading-function-prediction\": \"$PREDICTION_DATA_LOADING_FUNCTION_PATH\", \"target-function\": \"$TARGET_FUNCTION_SLUG\"}" "$DATA_YAML"

    # Update the most-recently-used-data-file
    yq -i ".most-recently-used-data-file = \"$DATA_PATH\"" "$DATA_YAML"

    echo "Unit '$UNIT_NAME' created successfully with data file: $DATA_PATH."
}

show_current_unit() {
    verify_data_file_exists
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
      echo "Data file: $(yq '.data_file' $FULL_CONFIG_PATH)"
      echo "Feedback file: $(yq '.feedback_file' $FULL_CONFIG_PATH)"
    else
      echo "BATCH LEARNING mode"
    fi

    echo ""
}

edit_current_unit() {
    verify_data_file_exists
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

verify_data_file_exists() {
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

add_function() {
  verify_data_file_exists
  verify_current_unit_is_set

  # Ensure arguments are provided
  FUNCTION_NAME="$1"
  if [ -z "$FUNCTION_NAME" ]; then
    echo "Usage: mdln function add <function-name>"
    exit 1
  fi

  # Retrieve config path for the current unit
  CONFIG_PATH=$(yq ".units.${CURRENT_UNIT}.config-path" $DATA_YAML)
  if [ -z "$CONFIG_PATH" ]; then
    echo "Error: Config path not found for the current unit '$CURRENT_UNIT'."
    exit 1
  fi

  FULL_CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_PATH"

  # Ensure the config file exists before modifying
  if [ ! -f "$FULL_CONFIG_PATH" ]; then
    echo "Error: Config file '$FULL_CONFIG_PATH' does not exist."
    exit 1
  fi

  # Construct function path and filename based on unit and function name
  FUNCTION_SLUG="/functions/$CURRENT_UNIT.$FUNCTION_NAME.py"
  FUNCTION_PATH="$DEFAULT_DATA_DIR$FUNCTION_SLUG"

  # Check for existing function with the same name
  CFN=$(yq ".units.${CURRENT_UNIT}.${FUNCTION_NAME}" "$DATA_YAML")
  if [ ! $CFN == "null" ]; then
    echo "Error: Function '$FUNCTION_NAME' already exists in the current unit's config."
    exit 1
  fi

  # Update unit config with function name and path
  # sed -i "/- name: $CURRENT_UNIT/a\    $FUNCTION_NAME: $FUNCTION_SLUG" "$DATA_YAML"
  yq -i ".units.\"${CURRENT_UNIT}\".\"${FUNCTION_NAME}\" = \"${FUNCTION_SLUG}\"" "$DATA_YAML"



  # Create a blank file for the new function
  touch "$FUNCTION_PATH"

  echo "Function '$FUNCTION_NAME' successfully added to unit '$CURRENT_UNIT'."
  echo "Placeholder file created at: $FUNCTION_PATH"
}

clean_unit_output() {
  # remove the model for the current unit
  MODEL_SLUG=$(yq ".units.${CURRENT_UNIT}.keras-filename" $DATA_YAML)
  DATA_DIR=$(yq ".data-directory" $DATA_YAML)

  echo "Giving you 10 seconds to come to your senses..."
  sleep 6

  echo "..4 seconds more.."
  sleep 2

  echo "..okay, here we go!"
  sleep 2

  rm $DATA_DIR$MODEL_SLUG

  yq -i ".units.\"${CURRENT_UNIT}\".keras-filename = null" "$DATA_YAML"

  echo "Done. Removed model for ${CURRENT_UNIT}"
}

list_functions() {
  verify_data_file_exists
  verify_current_unit_is_set

  yq eval ".units[\"$CURRENT_UNIT\"] | to_entries | map(select(.key | test(\"function$\"))) | from_entries" "$DATA_YAML"
}

remove_function() {
  verify_data_file_exists
  verify_current_unit_is_set

  # Ensure arguments are provided
  FUNCTION_NAME="$1"
  if [ -z "$FUNCTION_NAME" ]; then
    echo "Usage: mdln function remove <function-name>"
    exit 1
  fi

  # Retrieve config path for the current unit
  CONFIG_PATH=$(yq ".units.${CURRENT_UNIT}.config-path" $DATA_YAML)
  if [ -z "$CONFIG_PATH" ]; then
    echo "Error: Config path not found for the current unit '$CURRENT_UNIT'."
    exit 1
  fi

  FULL_CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_PATH"

  # Ensure the config file exists before modifying
  if [ ! -f "$FULL_CONFIG_PATH" ]; then
    echo "Error: Config file '$FULL_CONFIG_PATH' does not exist."
    exit 1
  fi

  # Construct function path and filename based on unit and function name
  FUNCTION_SLUG="/functions/$CURRENT_UNIT.$FUNCTION_NAME.py"
  FUNCTION_PATH="$DEFAULT_DATA_DIR$FUNCTION_SLUG"

  # Check for existing function with the same name
  CFN=$(yq ".units.${CURRENT_UNIT}.${FUNCTION_NAME}" "$DATA_YAML")
  if [ $CFN == "null" ]; then
    echo "Error: Function '$FUNCTION_NAME' does not exist in the current unit's config."
    exit 1
  fi

  # Update unit config with function name and path
  yq -i ".units.\"${CURRENT_UNIT}\".\"${FUNCTION_NAME}\" = null" "$DATA_YAML"

  echo "Function '$FUNCTION_NAME' successfully removed from unit '$CURRENT_UNIT'."

  if [ "$2" == "-f" ]; then
    rm $FUNCTION_PATH
    echo "Deleted the file $FUNCTION_PATH"
  fi
}

run_predictions() {
  python3 /home/jjames/src/learning/btcmodel/btcmodel/predicting/predict.py
}

run_training() {

  # Retrieve config path for the current unit
  CONFIG_PATH=$(yq ".units.${CURRENT_UNIT}.config-path" $DATA_YAML)
  if [ -z "$CONFIG_PATH" ]; then
    echo "Error: Config path not found for the current unit '$CURRENT_UNIT'."
    exit 1
  fi

  FULL_CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_PATH"

  # Ensure the config file exists before modifying
  if [ ! -f "$FULL_CONFIG_PATH" ]; then
    echo "Error: Config file '$FULL_CONFIG_PATH' does not exist."
    exit 1
  fi

  # Retrieve the value of USE_ONLINE_LEARNING_MODE from the YAML config
  USE_ONLINE_LEARNING_MODE=$(yq ".use_online_learning" $FULL_CONFIG_PATH)

  # Check if USE_ONLINE_LEARNING_MODE is set and equals "true" (case-insensitive)
  if [ "$USE_ONLINE_LEARNING_MODE" = "true" ] || [ "$USE_ONLINE_LEARNING_MODE" = "True" ]; then
    cd /home/jjames/src/learning/btcmodel/
    python3 -m btcmodel.training.online_learn.online_learn 
  else
    cd /home/jjames/src/learning/btcmodel/
    python3 -m btcmodel.training.batch.batch
    # python3 /home/jjames/src/learning/btcmodel/btcmodel/training/batch/batch.py
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
        run_training
        ;;
    function)
      SUBCOMMAND="$1"
      case "$SUBCOMMAND" in
        add)
          add_function "$2"
          ;;
        list)
          list_functions
          ;;
        remove)
          remove_function "$2" "$3"
          ;;
        *)
          echo "Usage: mdln function list|add|remove <function-name>"
          exit 1
          ;;
      esac
      ;;
    *)
        echo "Usage: mdln {init | list | new | use | show | edit | clean | predict | train | function add|list|remove}"
        exit 1
        ;;
esac

