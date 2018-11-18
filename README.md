# small-bang
In  the beginning there was the big-bang, that created everything and all machine learning ideas and algorithms. Now comes the small-bang, a reimplementation of (nearly) everything.

## Instalation
Just git clone and run `python3 -m pip install -r requirements.txt`

## Running
Just run whatever model you want to test, by default invocating it directly will run a simple test.
Make sure you run it from its root directory, for example, to run the K-NN algorithm, use `cd knn; python3 knn.py`

## Requeriments
Since I don't wan't to completely reinvent the wheel, I am frequently using functions from numpy, scipy and scikit learn. These are mostly things to make sure all the heavy numerical processing is done by [vectorizing](https://stackoverflow.com/questions/47755442/what-is-vectorization) numpy code, alongside loading datasets that come built-in in scikit-learn.
