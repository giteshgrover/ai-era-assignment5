from setuptools import setup, find_packages

setup(
    name="mnist-classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch --index-url https://download.pytorch.org/whl/cpu',
        'torchvision --index-url https://download.pytorch.org/whl/cpu',
        'pytest'
    ]
) 