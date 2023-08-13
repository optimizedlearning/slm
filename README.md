# Simple Small Language Model Training

This repo is an attempt to set up training of a language model on the pile
using a bunch of libraries: [huggingface](https://huggingface.co/), [wandb](https://wandb.ai), [pytorch lightning](https://lightning.ai/docs/pytorch/latest/) and [hydra](https://hydra.cc/)

The goal is to provide an example of how all these tools can be put together.
The model is a GPT model, with comments describing where the various architectural decision came from in the paper.

To run: if you are on BU's SCC, you can start with `source scc_setup.sh`. 

Then `python gpt_pile.py`.

You can submit a job on SCC for a longer run with `qsub slm_gpt2_train.sh`
