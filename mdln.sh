#!/bin/bash

# Maudlin Main Script: mdln

# Configurations
DEFAULT_DATA_DIR="$HOME/src/_data/maudlin"
DEFAULT_CONFIG_FILE="default.config.yaml"
DATA_YAML="$DEFAULT_DATA_DIR/maudlin.data.yaml"

# Functions
initialize_maudlin() {
    echo "Initializing Maudlin directory structure..."
    mkdir -p "$DEFAULT_DATA_DIR/configs" \
             "$DEFAULT_DATA_DIR/models" \
             "$DEFAULT_DATA_DIR/target_functions" \
             "$DEFAULT_DATA_DIR/inputs"
             
    # Git init for configs
    if [ ! -d "$DEFAULT_DATA_DIR/configs/.git" ]; then
        echo "Setting up git for configs..."
        git -C "$DEFAULT_DATA_DIR/configs" init
    fi

    # Create maudlin.data.yaml if it doesn't exist
    if [ ! -f "$DATA_YAML" ]; then
        echo "Creating maudlin.data.yaml..."
        echo "units: []" > "$DATA_YAML"
        echo "most-recently-used-data-file: null" >> "$DATA_YAML"
    fi

    echo "Maudlin initialization complete."
}

list_units() {
    echo "Listing all managed units:"
    if [ -f "$DATA_YAML" ]; then
        yq '.units | .[] | .name' "$DATA_YAML"  # Requires yq for YAML parsing
    else
        echo "Maudlin data file not found. Perhaps you need to initialize Maudlin first using 'mdln init'."
    fi
}

new_unit() {
    if [ ! -f "$DATA_YAML" ]; then
        echo "Maudlin data file not found. Perhaps you need to initialize Maudlin first using 'mdln init'."
        exit
    fi

    UNIT_NAME="$1"
    DATA_PATH="$2"

    # Ensure UNIT_NAME is provided
    if [ -z "$UNIT_NAME" ]; then
        echo "Usage: mdln new <name> [<data-path>]"
        exit 1
    fi

    # If DATA_PATH is not supplied, use the most-recently-used-data-file
    if [ -z "$DATA_PATH" ]; then
        DATA_PATH=$(yq '.most-recently-used-data-file' "$DATA_YAML")
        if [ "$DATA_PATH" == "null" ]; then
            echo "Error: No data-path supplied, and no most-recently-used-data-file is set in $DATA_YAML."
            exit 1
        fi
    fi

    # Ensure unique unit name
    if [[ -n "$(yq ".units | .[] | select(.name == \"$UNIT_NAME\")" "$DATA_YAML")" ]]; then
        echo "Unit name '$UNIT_NAME' already exists. Please choose a unique name."
        exit 1
    fi

    # Prepare new unit
    echo "Creating a new unit '$UNIT_NAME'..."
    CONFIG_PATH="$DEFAULT_DATA_DIR/configs/$UNIT_NAME.config.yaml"
    TARGET_FUNCTION_PATH="$DEFAULT_DATA_DIR/target_functions/$UNIT_NAME.target_function.py"

    # Copy the default config file to the new config
    cp "./$DEFAULT_CONFIG_FILE" "$CONFIG_PATH"
    echo "# Blank target function for $UNIT_NAME" > "$TARGET_FUNCTION_PATH"

    # Commit the config file
    pushd "$DEFAULT_DATA_DIR/configs" > /dev/null
    git add "$UNIT_NAME.config.yaml"
    git commit -m "Add config for $UNIT_NAME"
    CONFIG_COMMIT_ID=$(git rev-parse HEAD)
    popd > /dev/null

    # Add to maudlin.data.yaml
    yq -i ".units += [{\"name\": \"$UNIT_NAME\", \"config-commit-id\": \"$CONFIG_COMMIT_ID\", \"keras-filename\": null, \"data-filename\": \"$DATA_PATH\", \"target-function\": \"$TARGET_FUNCTION_PATH\"}]" "$DATA_YAML"

    # Update the most-recently-used-data-file
    yq -i ".most-recently-used-data-file = \"$DATA_PATH\"" "$DATA_YAML"

    echo "Unit '$UNIT_NAME' created successfully with data file: $DATA_PATH."
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
    *)
        echo "Usage: mdln {init|list|new}"
        exit 1
        ;;
esac

