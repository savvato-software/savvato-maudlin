
from setuptools import setup, find_packages

setup(
    name='maudlin',
    version='0.0.1',
    author='Johnathan James and ChatGPT',
    author_email='johnathan@savvato.com',
    description='Maudlin Framework for Neural Network Experimentation',
    url='https://github.com/savvato-software/savvato-maudlin',
    packages=find_packages(),
    install_requires=[
        'argparse',
        'scikit-learn',  # Replacing sklearn with its correct PyPI name
        'pyyaml',        # YAML library
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

