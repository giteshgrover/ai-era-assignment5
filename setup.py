from setuptools import setup, find_packages

setup(
    name="mnist-classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pytest'
    ]
) 