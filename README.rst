Recurrent Neural Network Grammars
+++++++++++++++++++++++++++++++++

This is an implementation of recurrent neural network grammars (RNNG) in PyTorch. RNNG is a neural constituency parser described in Dyer, C., Kuncoro, A., Ballesteros, M., & Smith, N. A. (2016). "Recurrent neural network grammars". In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 199–209)*. San Diego, California: Association for Computational Linguistics. Retrieved from http://www.aclweb.org/anthology/N16-1024. Dyer et al. also released their original DyNet implementation in https://github.com/clab/rnng. Please do check their work.

**CAUTION: This implementation is still a work in progress. There is absolutely no guarantee of backward compatibility or even this is going to work. Do not use this for real work!**

Contributing
============

Requirements
------------

Make sure you have installed:

#. Python 3.6
#. PyTorch 0.2. Please follow the installation instruction `here <http://pytorch.org/previous-versions/>`_. Note that the latest PyTorch version is 0.3 and here we need 0.2.

Next, install all the requirements in ``requirements.txt`` ::

    pip install -r requirements.txt

Then, install this package in development mode ::

    pip install -e .

Running the tests
-----------------

Run ``pytest`` from the project directory.

Running the linters
-------------------

Run ``flake8`` from the project directory. This will also run ``mypy`` to check type annotations, thanks to ``flake8-mypy``.

License
=======

MIT
