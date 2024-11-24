#!/bin/bash

# Maudlin Main Script: mdln

# Configurations
DEFAULT_DATA_DIR="$HOME/src/_data/maudlin"
DEFAULT_CONFIG_FILE="default.config.yaml"
DATA_FILE="$DEFAULT_DATA_DIR/maudlin.data.yaml"

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
    if [ ! -f "$DATA_FILE" ]; then
        echo "Creating maudlin.data.yaml..."
        echo "units: []" > "$DATA_FILE"
        echo "most-recently-used-data-file: null" >> "$DATA_FILE"
    fi

    # Copy the default config file if provided
    if [ ! -f "$DEFAULT_DATA_DIR/$DEFAULT_CONFIG_FILE" ]; then
        echo "Copying default config file..."
        cp "./$DEFAULT_CONFIG_FILE" "$DEFAULT_DATA_DIR/$DEFAULT_CONFIG_FILE"
    fi

    echo "Maudlin initialization complete."
}

list_units() {
    echo "Listing all managed units:"
    if [ -f "$DATA_FILE" ]; then
        yq '.units | .[] | .name' "$DATA_FILE"  # Requires yq for YAML parsing
    else
        echo "No units found. Initialize Maudlin first using 'mdln init'."
    fi
}

new_unit() {
    if [ $# -lt 2 ]; then
        echo "Usage: mdln new <name> <data-path>"
        exit 1
    fi

    UNIT_NAME="$1"
    DATA_PATH="$2"

    # Ensure unique unit name
    if yq ".units | .[] | select(.name == \"$UNIT_NAME\")" "$DATA_FILE" > /dev/null; then
        echo "Unit name '$UNIT_NAME' already exists. Please choose a unique name."
        exit 1
    fi

    # Prepare new unit
    echo "Creating a new unit '$UNIT_NAME'..."
    CONFIG_PATH="$DEFAULT_DATA_DIR/configs/$UNIT_NAME.config.yaml"
    TARGET_FUNCTION_PATH="$DEFAULT_DATA_DIR/target_functions/$UNIT_NAME.target_function.py"

    cp "$DEFAULT_DATA_DIR/$DEFAULT_CONFIG_FILE" "$CONFIG_PATH"
    echo "# Blank target function for $UNIT_NAME" > "$TARGET_FUNCTION_PATH"

    # Commit the config file
    pushd "$DEFAULT_DATA_DIR/configs" > /dev/null
    git add "$UNIT_NAME.config.yaml"
    git commit -m "Add config for $UNIT_NAME"
    CONFIG_COMMIT_ID=$(git rev-parse HEAD)
    popd > /dev/null

    # Add to maudlin.data.yaml
    yq -i ".units += [{\"name\": \"$UNIT_NAME\", \"config-commit-id\": \"$CONFIG_COMMIT_ID\", \"keras-filename\": null, \"data-filename\": \"$DATA_PATH\", \"target-function\": \"$TARGET_FUNCTION_PATH\"}]" "$DATA_FILE"
    echo "Unit '$UNIT_NAME' created successfully."
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

