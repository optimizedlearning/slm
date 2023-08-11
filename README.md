# Simple Small Language Model Training

This repo is an attempt to set up training of a language model on the pile
using a bunch of libraries: huggingface, wandb, pytorch lightning and hydra.
The goal is to provide an example of how all these tools can be put together.
The model is GPT3, with comments describing where the various architectural decision came from in the paper.

To run: if you are on BU's SCC, you can start with `source scc_setup.sh`. 

Then `python gpt_pile.py`.
