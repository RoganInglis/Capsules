# Dynamic Routing Between Capsules
TensorFlow implementation of [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).

## Requirements

Created and tested using:
- Python 3.5
- TensorFlow 1.4

#### Packages

- tensorflow
- numpy
- tqdm 
- matplotlib 

```commandline
pip install -r requirements.txt
```

Or for GPU TensorFlow:

```commandline
pip install -r requirements-gpu.txt
```

## Usage

To train a model on MNIST with default parameters:

```commandline
python main.py
```

To test a trained model:

```commandline
python main.py --test True --result_dir <result_dir>
```

Additional command line arguments are detailed in main.py.

## Results

Results on MNIST:

![MNIST Results](images/mnist_results.png?raw=true "MNIST Results")


## Notes

### TODO

- [ ] Refactor to be more consistent with Matrix Capsules with EM Routing implementation

## Reference

1. [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
2. [Project structure](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3)
