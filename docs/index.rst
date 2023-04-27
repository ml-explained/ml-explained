##############
ML Explained
##############

Welcome to ML Explained, an educational website to sharpen your Machine Learning knowledge!

Background
==========

Hi! We, Jordan and Akshil are co-authors of ML Explained, a resource that aims to educate those new to Data Science - specifically Machine Learning 
and Reinforcement Learning. We both are in the process of completing our PhDs at the University of Bath. During our time as PhD students we found that 
there was not a centralised pool of knowledge online. This has motivated us to attempt to fill in the gaps with tutorial-like posts where we implement 
various concepts and algorithms all here!

We have a few tutorial channels, each with a different theme or purpose.

How to: Machine Learning
========================

A beginner's guide to machine learning designed for those that know a little Python and some key terms. Suited for those in education who want to 
understand the algorithms. A **from scratch** attitude is adopted here whereby most things will be built using **numpy** and **scipy** instead of importing 
off-the-shelf algorithms from **sklearn** (scikit-learn).


How to: Reinforcement Learning
==============================

An introduction to Reinforcement Learning starting from what it is, and going from the basics with Markov Decision Processes to function approximation 
with Neural Networks. Tabular methods are implemented in **numpy** whilst function approximation is in **PyTorch**.


Advanced Applications
=====================

For those that are already well-versed in the field, this channel builds further on the **How to** tutorials series, applying ML and RL techniques on 
more interesting and more realistic problems.

Theories
========

For those that want the more formal derivations, this channel aims to equip you with the Mathematical knowledge to understand the main assumptions 
behind these Data Science concepts.

Installation
============

If you would like to access our datasets or execute our code on your problems, first make sure you have ``git`` and ``pip`` readily available within 
your terminal then execute the following

.. code-block:: text

    # optionally create a virtual environment
    python3 -m venv env

    # activate virtual environment
    # linux / macos
    source env/bin/activate

    # windows
    .\env\Scripts\activate

    # install from our GitHub repository
    pip install -e git+https://github.com/ml-explained/ml-explained.git#egg=ml_explained


.. toctree::
    :hidden:

    Home <self>


.. toctree::
    :hidden:
    :caption: How to: Machine Learning

    how-to-machine-learning/index.rst

.. toctree::
    :hidden:
    :caption: How to: Reinforcement Learning

    how-to-reinforcement-learning/index.rst

.. toctree::
    :hidden:
    :caption: Advanced Applications
    
    advanced-applications/index.rst

.. toctree::
    :hidden:
    :titlesonly:
    :caption: Theories

    theories/linear-optimisation/overview.rst

