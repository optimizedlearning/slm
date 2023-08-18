import composer
from composer.loggers.logger import Logger
from composer.core import State
import hashlib


class WandBLoggerWithAutoResume(composer.loggers.WandBLogger):
    # Enables autoresume in the wandb logger by making the ID be a deterministic
    # function of the run_name. Thus, but setting the same run_name as a previous
    # run you can simultaneously enable autoresume of checkpoints and connect to the
    # same wandb run.
    # Note that if you change the name in the wandb web UI, you can still find
    # the original name as it is logged under model.run_name in the run attributes.

    def __init__(self, *args, resume=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume = resume

    def init(self, state: State, logger: Logger) -> None:
        # loggers are subclasses of Callback.i
        # https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html#custom-callbacks
        # This function is called during the __init__ method of Trainer,
        # although it is called *before* training loads any checkpoints.

        # compute the run_id as a function of the run_name.
        # ids must be globally unique, but we are very unlikely to ever
        # reuse a run name.
        run_id = hashlib.md5(state.run_name.encode("ascii")).hexdigest()
        print(f"state run name: {state.run_name}, run_id: {run_id}")
        if "id" not in self._init_kwargs:
            # if for some reason the user has manually specified the id,
            # let us respect that choice.
            self._init_kwargs["id"] = run_id

        # bit hacky here: I don't want to force the wandb run to have the
        # same name as the composer name, so we need to temporarily delete the
        # name from the state here.
        # This is basically just to make sure that if the user has changed the
        # name in the wandb web ui, then we are not going to overwrite it here.
        if self.resume:
            run_name = state.run_name
            state.run_name = None
        super().init(state, logger)
        if self.resume:
            state.run_name = run_name
