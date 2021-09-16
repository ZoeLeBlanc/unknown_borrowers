# Unknown Borrowers recommender

This directory contains a python recommender script for [Shakespeare and Company Project](https://shakespeareandco.princeton.edu/) data implemented with the [lightfm python library](https://github.com/lyst/lightfm).

## setup

Install the python packages needed for the recommender script:

```sh
pip install numpy pandas lightfm rich
```

Once dependencies are installed, you can run the script::

```sh
python shxco_recommender.py
```

To check the performance of the model, run the validation script::

```sh
python validation.py
```

For convenience, the 1.1 versions of Shakespeare and Company Project data are included in the data folder. The path in the script assumes you run it from this directory.

