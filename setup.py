import os
from setuptools import setup, find_packages

print(find_packages())

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ml-explained",
    version = "0.0.1",
    author = "Jordan Taylor, Akshil Patel",
    description = "",
    license = "BSD",
    url = "https://github.com/ml-explained/ml-explained.git",
    packages = ['ml_explained'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    
    ],
    install_requires = read('requirements.txt')
)