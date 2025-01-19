import pytest
from src.scenario.diff_to_sed_commands import diff_to_sed_commands

def test_diff_to_sed_commands():
    diff = [
        "--- /home/jjames/src/_data/maudlin/configs/bank-csv2.config.yaml",
        "+++ /tmp/tmpnv1podq7.txt",
        "@@ -70,7 +70,7 @@",
        "   - {layer_type: Dropout, rate: 0.2300449301547453}",
        "   - {activation: sigmoid, layer_type: Dense, units: 1}",
        "   optimization:",
        "-    activation: [relu, leaky_relu]",
        "+    activation: [relu, leaky_relui, swish]",
        "     batch_size: {max: 64, min: 8}",
        "     delta: 0.001",
        "     dropout: {max: 0.35, min: 0.1}"
    ]

    expected_sed_commands = [
        "/^   optimization:/,/^   \\S/{s/activation: \\[relu, leaky_relu\\]/d/}",
        "/^   optimization:/a\\    activation: [relu, leaky_relui, swish]"
    ]

    result = diff_to_sed_commands(diff)
    assert result == expected_sed_commands