
import os
import sys
import yaml
import json
import argparse
from datetime import datetime
from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.layout import Layout
from rich.panel import Panel
from difflib import unified_diff
import termios
import tty

from maudlin_core.src.lib.framework.maudlin import load_maudlin_data

console = Console()

# Load Maudlin data
maudlin = load_maudlin_data()

# Load history.yaml
def load_history():
    history_path = os.path.join(
        maudlin['data-directory'], 'trainings', maudlin['current-unit'], 'history.yaml')
    with open(history_path, "r") as f:
        return yaml.safe_load(f)

# Load best_metrics.json for specific runs
def load_best_metrics(run_id):
    metrics_path = os.path.join(
        maudlin['data-directory'], 'trainings', maudlin['current-unit'], f'run_{run_id}', 'best_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    
    raise ValueError(f"best_metrics.json for run {run_id} was not found.")

# Load config.yaml for specific runs
def load_config(run_id):
    config_path = os.path.join(
        maudlin['data-directory'], 'trainings', maudlin['current-unit'], f'run_{run_id}', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return None

# Get key press without root access
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)  # Read one byte first
        if key == '\x1b':       # Check if it's an escape sequence
            key += sys.stdin.read(2)  # Read the remaining two bytes
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

# Display detailed run information
def clean_config_diff(raw_diff):
    """Clean and format the raw diff string for display."""
    # Split the raw diff into lines
    lines = raw_diff.splitlines()
    
    # Find the actual diff lines (ignore metadata headers)
    diff_start = 0
    for i, line in enumerate(lines):
        if line.startswith('@@'):
            diff_start = i - 1  # Include the first metadata header for context
            break

    # Extract meaningful diff lines
    meaningful_diff = lines[diff_start:] if diff_start > 0 else lines

    # Join and return the clean diff
    return "\n".join(meaningful_diff)

# Load classification report
def load_classification_report(run_id):
    path = os.path.join(
        maudlin['data-directory'], 'trainings', maudlin['current-unit'], f'run_{run_id}',
        'post_training/classification_report.txt'
    )
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


# Load correlation metrics
def load_correlation_metrics(run_id):
    path = os.path.join(
        maudlin['data-directory'], 'trainings', maudlin['current-unit'], f'run_{run_id}',
        'pre_training/correlation_metrics.json'
    )
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# Display correlation metrics
def display_correlation_metrics(metrics):
    console.print("\n[bold cyan]Correlation Metrics:[/]")
    console.print(f"Max Correlation: {metrics['max_correlation'][0]} and {metrics['max_correlation'][1]} - {metrics['max_correlation'][2]:.4f}")
    console.print(f"Min Correlation: {metrics['min_correlation'][0]} and {metrics['min_correlation'][1]} - {metrics['min_correlation'][2]:.4f}")
    console.print(f"Average Correlation: {metrics['avg_correlation']:.6f}")

    console.print("\n[bold cyan]Top 5 Highest Correlations:[/]")
    table = Table(title="Top 5 Correlations", show_header=True, header_style="bold green")
    table.add_column("Feature 1", style="cyan")
    table.add_column("Feature 2", style="cyan")
    table.add_column("Value", justify="right")

    for pair in metrics['top_5_highest_correlations']:
        table.add_row(pair[0], pair[1], f"{pair[2]:.4f}")

    console.print(table)

    console.print("\n[bold cyan]Bottom 5 Lowest Correlations:[/]")
    table = Table(title="Bottom 5 Correlations", show_header=True, header_style="bold red")
    table.add_column("Feature 1", style="cyan")
    table.add_column("Feature 2", style="cyan")
    table.add_column("Value", justify="right")

    for pair in metrics['bottom_5_lowest_correlations']:
        table.add_row(pair[0], pair[1], f"{pair[2]:.4f}")

    console.print(table)


def build_metrics_panel(metrics):
    """Build metrics panel."""
    table = Table(title="Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    if metrics:
        for k, v in metrics.items():
            table.add_row(k, f"{v['best_value']:.6f}")
    else:
        table.add_row("[yellow]No metrics available.[/]", "")
    return Panel(table, title="[bold cyan]Metrics[/]")


def build_config_changes_panel(config_diff):
    """Build config changes panel."""
    if config_diff:
        formatted_diff = clean_config_diff(config_diff)
        syntax = Syntax(formatted_diff, "diff", theme="monokai", line_numbers=False)
        return Panel(syntax, title="[bold cyan]Config Changes[/]")
    return Panel("[yellow]No configuration changes.[/]", title="[bold cyan]Config Changes[/]")


def build_correlation_panel(metrics):
    """Build correlation metrics panel."""
    table = Table(title="Correlation Metrics", show_header=True, header_style="bold blue")
    table.add_column("Feature 1", style="cyan")
    table.add_column("Feature 2", style="cyan")
    table.add_column("Value", justify="right")

    for pair in metrics['top_5_highest_correlations']:
        table.add_row(pair[0], pair[1], f"{pair[2]:.4f}")

    for pair in metrics['bottom_5_lowest_correlations']:
        table.add_row(pair[0], pair[1], f"{pair[2]:.4f}")

    return Panel(table, title="[bold cyan]Correlation Metrics[/]")


def build_classification_report_panel(report):
    """Build classification report panel."""
    if report:
        syntax = Syntax(report, "plaintext", theme="monokai", line_numbers=False)
        return Panel(syntax, title="[bold cyan]Classification Report[/]")
    return Panel("[yellow]No classification report available.[/]", title="[bold cyan]Classification Report[/]")


def display_run_details(run, metrics, config):
    """Display run details in a structured layout."""
    console.clear()

    # Load data
    correlation_metrics = load_correlation_metrics(run['id'])
    classification_report = load_classification_report(run['id'])

    # Build panels
    metrics_panel = build_metrics_panel(metrics)
    config_changes_panel = build_config_changes_panel(run.get('config_diff', ""))
    correlation_panel = build_correlation_panel(correlation_metrics) if correlation_metrics else Panel("[yellow]No data[/]")
    classification_panel = build_classification_report_panel(classification_report)

    # Create layout
    layout = Layout()

    # Top half
    layout.split_row(
        Layout(name="left"), Layout(name="right")
    )
    layout["left"].split(Layout(metrics_panel), Layout(correlation_panel))
    layout["right"].split(Layout(config_changes_panel), Layout(classification_panel))

    # Render layout
    console.print(layout)

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
import time

# Select a child from a list
def select_child(children):
    current_index = 0
    while True:
        console.clear()
        table = Table(title="Select a Child", header_style="bold cyan")
        table.add_column("Index")
        table.add_column("Child ID")

        for idx, child_id in enumerate(children):
            if idx == current_index:
                table.add_row(f"[bold yellow]{idx} ->[/]", str(child_id))
            else:
                table.add_row(str(idx), str(child_id))

        console.print(Panel(table, title="[bold cyan]Child Selection[/]"))

        key = get_key()
        if key in ('k', '\x1b[A') and current_index > 0:
            current_index -= 1
        elif key in ('j', '\x1b[B') and current_index < len(children) - 1:
            current_index += 1
        elif key == '\r':
            return children[current_index]

def interactive_view(history):
    runs = {run['id']: run for run in history['history']}
    current_id = 1
    current_tab = 0  # 0 for first tab, 1 for second tab
    fullscreen_panel = None  # Track if a panel is in fullscreen mode

    # Render the layout with selected tab
    def render_view(current_id, current_tab):
        run = runs[current_id]
        metrics = load_best_metrics(run['id'])
        config = load_config(run['id'])
        correlation_metrics = load_correlation_metrics(run['id'])
        classification_report = load_classification_report(run['id'])

        # Load currently selected run ID
        metadata_path = os.path.join(
            maudlin['data-directory'], 'trainings', maudlin['current-unit'], 'run_metadata.json'
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            selected_run_id = metadata.get('current_run_id', 'None')
        else:
            selected_run_id = 'None'

        # Build panels
        metrics_panel = build_metrics_panel(metrics)
        config_changes_panel = build_config_changes_panel(run.get('config_diff', ""))
        correlation_panel = build_correlation_panel(correlation_metrics) if correlation_metrics else Panel("[yellow]No data[/]", title="Correlation Metrics")
        classification_panel = build_classification_report_panel(classification_report)

        # If in fullscreen mode, show only the selected panel
        if fullscreen_panel == 1:
            return Layout(metrics_panel)
        elif fullscreen_panel == 2:
            return Layout(config_changes_panel)
        elif fullscreen_panel == 3:
            return Layout(classification_panel)
        elif fullscreen_panel == 4:
            return Layout(correlation_panel)

        # Create the layout
        layout = Layout()

        # Add a header row showing both run IDs
        layout.split_column(
            Layout(Panel(""), size=1),  # Blank row
            Layout(Panel(f"[bold]Current Run ID:[/] {current_id}  [bold]Selected Run ID:[/] {selected_run_id}", title="Header"), size=3),
            Layout(name="main")
        )

        # Tab-specific layouts
        if current_tab == 0:
            # First tab: Metrics (Top) and Config Changes (Bottom, takes remaining space)
            layout["main"].split_column(
                Layout(metrics_panel, size=10),  # Fixed height for metrics
                Layout(config_changes_panel)
            )
        elif current_tab == 1:
            # Second tab: Classification Report (Top) and Correlation Metrics (Bottom, takes remaining space)
            layout["main"].split_column(
                Layout(classification_panel, size=10),  # Fixed height for classification report
                Layout(correlation_panel)
            )

        return layout

    previous_child = {}
    cleared_selection = set()

    while True:
        # Render the current view
        console.clear()
        console.print(render_view(current_id, current_tab))

        key = get_key()
        run = runs[current_id]
        if key.isdigit() and int(key) > 0:
            # Toggle fullscreen for a panel
            panel_index = int(key)
            if fullscreen_panel == panel_index:
                fullscreen_panel = None  # Exit fullscreen mode
            else:
                fullscreen_panel = panel_index
        elif fullscreen_panel is None:  # Normal mode
            if key in ('k', '\x1b[A'):
                # Move up to the parent run
                if run['parent']:
                    previous_child[run['parent']] = current_id  # Remember the selected child
                    current_id = run['parent']
                else:
                    cleared_selection.add(current_id)  # Clear selection for root node
            elif key in ('j', '\x1b[B') and run['children']:
                # Move down to a child run
                if len(run['children']) == 1:
                    current_id = run['children'][0]
                elif current_id in previous_child and current_id not in cleared_selection:
                    current_id = previous_child[current_id]  # Use previously selected child
                else:
                    current_id = select_child(run['children'])
                    cleared_selection.discard(current_id)  # Reset cleared state after selection
            elif key in ('h', '\x1b[D'):
                # Switch to the previous tab
                current_tab = (current_tab - 1) % 2
            elif key in ('l', '\x1b[C'):
                # Switch to the next tab
                current_tab = (current_tab + 1) % 2
            elif key == 'q':
                # Quit interactive mode
                break


# Function to update the selected run ID in run_metadata.json
def update_selected_run_id(run_id):
    metadata_path = os.path.join(
        maudlin['data-directory'], 'trainings', maudlin['current-unit'], 'run_metadata.json'
    )

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata['current_run_id'] = run_id

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # copy the config for current run to the default data directory
        config_path = os.path.join(
            maudlin['data-directory'], 'trainings', maudlin['current-unit'], f'run_{run_id}', 'config.yaml')
        dest_path = os.path.join(
            maudlin['data-directory'], 'config.yaml'
        )

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        with open(dest_path, 'w') as f:
            yaml.safe_dump(config, f)
    else:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")


def tree_view(history):
    pass

def list_view(history):
    pass


def main():

    # Argument Parser Setup
    parser = argparse.ArgumentParser(description="Visualize Training History")

    # Positional argument for the history file path
    parser.add_argument('history_file', type=str, help='Path to history file')

    # Optional flags
    parser.add_argument('-i', '--interactive', action='store_true', help='Scroll through training runs')
    parser.add_argument('-t', '--tree', action='store_true', help='Hierarchical view of training runs')
    parser.add_argument('-l', '--list', action='store_true', help='List training runs')

    # Parse Arguments
    args = parser.parse_args()

    # Load history
    history = load_history()

    # Handle flags
    if args.interactive:
        interactive_view(history)
    elif args.tree:
        tree_view(history)
    elif args.list:
        list_view(history)
    else:
        print("No view mode specified. Use --interactive, --tree, or --list.")

if __name__ == "__main__":
    main()

