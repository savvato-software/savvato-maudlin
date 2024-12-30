from setuptools import setup, find_packages

setup(
    name="maudlin",  # Package name
    version="0.0.1",  # Version
    packages=find_packages(),  # Automatically find all packages
    include_package_data=True,  # Include non-code files like YAML configs
    install_requires=[
        # List dependencies here (e.g., numpy, pandas)
        # Example:
        # 'numpy>=1.21.0',
    ],
    entry_points={
        'console_scripts': [
            'mdln=maudlin_core.cli:main',  # Maps 'mdln' command to 'main()' in cli.py
        ],
    },
    python_requires=">=3.6",  # Minimum Python version
)

