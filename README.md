# Simple Small Language Model Training

This repo is an attempt to set up training of a language model on the pile
using a bunch of libraries: [huggingface](https://huggingface.co/), [wandb](https://wandb.ai), [pytorch lightning](https://lightning.ai/docs/pytorch/latest/), [composer](https://docs.mosaicml.com/projects/composer/en/stable/index.html), [streaming](https://github.com/mosaicml/streaming) and [hydra](https://hydra.cc/)

The goal is to provide an example of how all these tools can be put together.
The model is a GPT model, with comments describing where the various architectural decision came from in the paper.

To run: if you are on BU's SCC, you can start with `source scc_setup.sh`. 

Then `python gpt_pile_composer.py`. or `python gpt_pile_lightning.py`.

You can submit a job on SCC for a longer run with `qsub slm_gpt2_train_composer.sh`.

You can run one with 2 gpus with `composer -n 2 gpt_pile_composer.py` or `qsub slm_gpt2_train_composer_2gpu.sh`.


The composer implementation is the most fully-featured right now. So far, it can:

* log more data automatically
* resume runs (just provide the `run_name=...` argument to be the same as the run you want to resume)
* automatically save checkpoints

The resuming feature currently "spins" the dataloader to find get back to the correct example. This
could probably be improved as the streaming dataset class is specifically designed to support random access
rather than just iteration.