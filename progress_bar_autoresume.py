import composer
from composer.core import State
from composer.loggers.logger import Logger
# The built-in progress bar loader only initializes the progress bar
# at the start of an epoch. If we are resuming from checkpoint then
# the start of the epoch has "already happened" and it doesn't get
# initialized. This class is a wrapper that makes sure the appropriate
# progress bar is initialized before starting any batch.

class ProgressBarWithAutoResume(composer.loggers.ProgressBarLogger):

    def batch_start(self, state: State, logger: Logger) -> None:
        super().batch_start(state, logger)
        if self.show_pbar and not self.train_pbar:
            self.train_pbar = self._build_pbar(state=state, is_train=True)

    def eval_batch_start(self, state: State, logger: Logger) -> None:
        super().eval_batch_start(state, logger)
        if self.show_pbar and not self.eval_pbar:
            self.eval_pbar = self._build_pbar(state=state, is_train=False)
