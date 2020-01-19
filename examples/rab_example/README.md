# Train and eval example for Reverse Angry Birds

## TOC
* What is Reverse Angry Birds?
* Usage
    * Examples
* Requirements

## What is Reverse Angry Birds(RAB)?

## Usage

Clone the repo

    git clone https://github.com/annoyedrussian/tf-unity
    cd tf-unity

`tf-unity.py` is the original test file. It should run a simple training loop for dqn agent on RAB environment. You will need to provide a path to Unity executable.

### Examples

More sophisticated example of RAB is located in <i>examples/rab_example</i>.

Make sure to add repo folder to PYTHONPATH environment variable

    export PYTHONPATH=<full_path_to_tf-unity_root_folder>

    cd examples/rab_example
    python3 train_eval.py

## Requirements

```ml-agents==0.13.0```

```gym-unity==0.13.0```