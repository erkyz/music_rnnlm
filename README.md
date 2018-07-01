# Neural generation of self-similar music without explicit constraints

## Requirements
* Python 2
* Pytorch 0.2
* [music21](http://web.mit.edu/music21/) (pip install music21)

## High-level steps
1) Preprocess data, or synthesize data. (See [process_data](./process_data))
2) Use either [sh/local.sh](./sh/local.sh) or [sh/sbatch.sh](./sh/sbatch.sh) to train a model and then generate from it. For example, from the top-level directory, you can run ```sh sh/local.sh base test_output.out gen_out "music_data/CMaj_Nottingham"```. Refer to all possible flags and hyperparams within the shell files and especially in [main.py](./main.py)! If running on a machine with Singularity and slurm, use [sh/sbatch.sh](./sh/sbatch.sh). This was designed to run on the CMU LTI tir cluster, but it can be easily modified to run elsewhere.

### A note on "channels"
Files were written to support having factorized embeddings, such that an event is split into multiple components, each with a separate vocabulary/dictionary. For example, we might want to factorize the embeddings for pitch and duration. For this reason, many objects are wrapped in lists so that multiple "channels" can be passed at once. HOWEVER, I have not run any code that actually uses factorized embeddings in a long time, so much of the code only supports having one channel.
