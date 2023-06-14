from importlib_metadata import entry_points
from setuptools import setup, find_packages

setup(
    name='simpsons_widescreen',
    version='0.0.1',
    entry_points={
        "console_scripts": [
            "train_widescreen = simpsons_widescreen.widescreen:main"
        ]
    },
    packages=find_packages(),
    install_requires=[
    ],
    zip_safe=False
)