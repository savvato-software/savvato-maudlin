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
        echo "data-directory: $DEFAULT_DATA_DIR" >> "$DATA_YAML"
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

set_current_unit() {
    if [ ! -f "$DATA_YAML" ]; then
        echo "Maudlin data file not found. Perhaps you need to initialize Maudlin first using 'mdln init'."
        exit 1
    fi

    UNIT_NAME="$1"

    # Ensure UNIT_NAME is provided
    if [ -z "$UNIT_NAME" ]; then
        echo "Usage: mdln use <name>"
        exit 1
    fi

    # Check if the unit exists
    UNIT_EXISTS=$(yq ".units | .[] | select(.name == \"$UNIT_NAME\")" "$DATA_YAML")
    if [ -z "$UNIT_EXISTS" ]; then
        echo "Error: Unit name '$UNIT_NAME' does not exist. Use 'mdln list' to see available units."
        exit 1
    fi

    # Update the current unit in maudlin.data.yaml
    yq -i ".current-unit = \"$UNIT_NAME\"" "$DATA_YAML"
    echo "Current unit set to '$UNIT_NAME'."
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

    ln -s $DATA_PATH "$DEFAULT_DATA_DIR/inputs/$UNIT_NAME-data"

    # Ensure unique unit name
    if [[ -n "$(yq ".units | .[] | select(.name == \"$UNIT_NAME\")" "$DATA_YAML")" ]]; then
        echo "Unit name '$UNIT_NAME' already exists. Please choose a unique name."
        exit 1
    fi

    # Prepare new unit
    echo "Creating a new unit '$UNIT_NAME'..."
    CONFIG_SLUG="/configs/$UNIT_NAME.config.yaml"
    CONFIG_PATH="$DEFAULT_DATA_DIR$CONFIG_SLUG"

    TARGET_FUNCTION_SLUG="/target_functions/$UNIT_NAME.target_function.py"
    TARGET_FUNCTION_PATH="$DEFAULT_DATA_DIR/$TARGET_FUNCTION_SLUG"

    # Copy the default config file to the new config
    cp "./$DEFAULT_CONFIG_FILE" "$CONFIG_PATH"
    sed -i "s|^data_file:.*|data_file: $DATA_PATH|" "$CONFIG_PATH"
    echo "# Blank target function for $UNIT_NAME" > "$TARGET_FUNCTION_PATH"

    # Commit the config file
    pushd "$DEFAULT_DATA_DIR/configs" > /dev/null
    git add "$UNIT_NAME.config.yaml"
    git commit -m "Add config for $UNIT_NAME"
    CONFIG_COMMIT_ID=$(git rev-parse HEAD)
    popd > /dev/null

    # Add to maudlin.data.yaml
    yq -i ".units += [{\"name\": \"$UNIT_NAME\", \"config-commit-id\": \"$CONFIG_COMMIT_ID\", \"config-path\": \"$CONFIG_SLUG\", \"keras-filename\": null, \"data-filename\": \"/inputs/${UNIT_NAME}-data\", \"target-function\": \"$TARGET_FUNCTION_SLUG\"}]" "$DATA_YAML"

    # Update the most-recently-used-data-file
    yq -i ".most-recently-used-data-file = \"$DATA_PATH\"" "$DATA_YAML"

    echo "Unit '$UNIT_NAME' created successfully with data file: $DATA_PATH."
}

show_current_unit() {
    if [ ! -f "$DATA_YAML" ]; then
        echo "Maudlin data file not found. Perhaps you need to initialize Maudlin first using 'mdln init'."
        exit 1
    fi

    # Check if the current unit is set
    CURRENT_UNIT=$(yq '.current-unit' "$DATA_YAML")
    if [ "$CURRENT_UNIT" == "null" ]; then
        echo "No current unit is set. Use 'mdln set <name>' to set a current unit."
        exit 1
    fi

    # Retrieve and display properties of the current unit
    UNIT_PROPERTIES=$(yq ".units | .[] | select(.name == \"$CURRENT_UNIT\")" "$DATA_YAML")
    if [ -z "$UNIT_PROPERTIES" ]; then
        echo "Error: Current unit '$CURRENT_UNIT' not found in units list. Check your maudlin.data.yaml file."
        exit 1
    fi

    echo "Current Unit: $CURRENT_UNIT"
    echo "Properties:"
    echo "$UNIT_PROPERTIES"
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
    *)
        echo "Usage: mdln {init | list | new | use | show}"
        exit 1
        ;;
esac

