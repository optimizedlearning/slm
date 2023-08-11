
from lightning.pytorch.callbacks import Callback
import time

class NoOpContext:
    def __init__(self):
        pass
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, exc_tb):
        return False


class LogTrainMetrics(Callback):

    def __init__(self):
        super().__init__()

        self.current_time = None
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, loss_data, batch, batch_idx):
        pl_module.log_dict(
            {
                "train/loss": loss_data["loss"],
                "train/accuracy": loss_data["accuracy"],
            }
        )
        if 'bits_per_byte' in loss_data:
            pl_module.log('train/bits_per_byte', loss_data['bits_per_byte'])
            pl_module.log('train/compression_ratio', loss_data['bits_per_byte']/8)

        if self.current_time is not None:
            prev_time = self.current_time
            self.current_time = time.time()
            delta = self.current_time - prev_time

            pl_module.log('train/batch_per_sec', 1.0/delta)
            pl_module.log('train/sec_per_batch', delta)
        else:
            self.current_time=time.time()


        total_sec = self.current_time - self.start_time
        pl_module.log('total_sec', total_sec)
        pl_module.log('total_min', total_sec/60)
        pl_module.log('total_hr', total_sec/(60*60))
        pl_module.log('avg_global_step_per_sec', pl_module.global_step/total_sec)
