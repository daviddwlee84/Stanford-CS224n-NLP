# Question Answering on SQuAD 2.0

## Setup

1. Make sure you have [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `source activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code
  
4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.

### Common Usage

* Train `python train.py -n <name>`
  * this will generate log `log.txt`, tensorboard tfevents `events.out.tfevents.*` and checkpoint `step_N.pth.tar`, `best.pth.tar`
* Test `python test.py -n <name> --load_path save/train/<name>-<ID>/best.pth.tar`
  * need to assign name `-n` or `--name`
  * need to assign `--load_path` (just look up in the `save/train/<model name>-<ID>/best.pth.tar`)
* Tracking progress in TensorBoard
  * remote/local `tensorboard --logdir save --port 8889`
    * `http://localhost:8889`
  * port forwarding to local port 1234 `ssh -N -f -L localhost:1234:localhost:8889 <user>@<remote>`
    * `http://localhost:1234`

> `ssh -NfL 1234:localhost:8889 <user>@<remote> > /dev/null 2&>1`

## Progress

1. [X] BiDAF without character-level embedding layer (default baseline)
   * Last Train (30 epoches): `Dev NLL: 03.28, F1: 59.77, EM: 56.41, AvNA: 66.93`
   * Test: `Dev NLL: 03.22, F1: 59.96, EM: 56.70, AvNA: 66.95`
   * Common Command
     * Train: `python train.py -n baseline -m BiDAF-baseline`
2. [ ] BiDAF-No-Answer (single model)
   * Common Command
     * Train: `python train.py -n BiDAF-No-Answer -m BiDAF-w-char`
   * Remarks
     * I used Assignment 5 char model embedding module and share half `hidden_size` with original word embedding then concatenate them together as the new word representation
3. [ ] ...(more improvement)

## Files

* `util.py`
  * `class CheckpointSaver`

> (TODO)

## Related Links

### Models

#### BiDAF

* [BiDAF](https://allenai.github.io/bi-att-flow/)
  * [code](https://github.com/allenai/bi-att-flow)
  * [pdf](https://arxiv.org/abs/1611.01603)
  * [demo](http://allgood.cs.washington.edu:1995/)
* [An Illustrated Guide to Bi-Directional Attention Flow (BiDAF)](https://towardsdatascience.com/the-definitive-guide-to-bi-directional-attention-flow-d0e96e9e666b)
