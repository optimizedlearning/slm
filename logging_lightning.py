
from lightning.pytorch.callbacks import Callback
import time


class LogTrainMetrics(Callback):

    def __init__(self):
        super().__init__()

        self.current_time = None
        self.start_time = time.time()

    def on_validation_batch_end(self, trainer, pl_module, loss_data, batch, batch_idx):
        self.on_any_batch_end('valid', trainer, pl_module, loss_data, batch, batch_idx)
    def on_train_batch_end(self, trainer, pl_module, loss_data, batch, batch_idx):
        self.on_any_batch_end('train', trainer, pl_module, loss_data, batch, batch_idx)

    # def on_after_backward(self, trainer, pl_module) -> None:
    #     print("on_after_backward enter")
    #     print("len trainable params: ",len(list(pl_module.named_parameters())))
    #     for name, p in pl_module.named_parameters():
    #         if p.grad is None:
    #             print(f"no gradient: {name}")
    #     print("on_after_backward exit")

    def on_any_batch_end(self, prefix, trainer, pl_module, loss_data, batch, batch_idx):
        sync_dist = (prefix == 'valid')
        pl_module.log_dict(
            {
                f"{prefix}/loss": loss_data["loss"],
                f"{prefix}/accuracy": loss_data["accuracy"],
            },
            sync_dist=sync_dist
        )

        if 'bits_per_byte' in loss_data:
            pl_module.log(f'{prefix}/bits_per_byte', loss_data['bits_per_byte'], sync_dist=sync_dist)
            pl_module.log(f'{prefix}/compression_ratio', loss_data['bits_per_byte']/8, sync_dist=sync_dist)

        if self.current_time is not None:
            prev_time = self.current_time
            self.current_time = time.time()
            delta = self.current_time - prev_time

            pl_module.log(f'{prefix}/batch_per_sec', 1.0/delta, sync_dist=sync_dist)
            pl_module.log(f'{prefix}/sec_per_batch', delta, sync_dist=sync_dist)
        else:
            self.current_time=time.time()

        total_sec = self.current_time - self.start_time
        pl_module.log('total_sec', total_sec, sync_dist=sync_dist)
        pl_module.log('total_min', total_sec/60, sync_dist=sync_dist)
        pl_module.log('total_hr', total_sec/(60*60), sync_dist=sync_dist)
        pl_module.log('avg_global_step_per_sec', pl_module.global_step/total_sec, sync_dist=sync_dist)


    
