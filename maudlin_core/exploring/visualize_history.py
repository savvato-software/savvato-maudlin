
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

from maudlin import load_maudlin_data

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
        key = sys.stdin.read(1)
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


# Interactive view without requiring root access
def interactive_view(history):
    runs = {run['id']: run for run in history['history']}
    current_id = 1

    while True:
        # Load details for the current run
        run = runs[current_id]
        metrics = load_best_metrics(run['id'])
        config = load_config(run['id'])
        display_run_details(run, metrics, config)

        # Handle user input
        key = get_key()
        if key == 'k':  # Move up
            parent_id = run['parent']
            if parent_id is not None:
                current_id = parent_id
        elif key == 'j':  # Move down
            if run['children']:
                current_id = run['children'][0]
        elif key == 'q':  # Quit
            break

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

