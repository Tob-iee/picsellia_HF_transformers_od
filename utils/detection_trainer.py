import logging
import os

from picsellia import Experiment
from picsellia.types.enums import LogType
from transformers import TrainerCallback, TrainingArguments

from utils.processing import send_run_to_picsellia

class PicselliaDetectionTrainer(TrainerCallback):
# class CustomPicselliaCallback(TrainerCallback):

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def on_train_begin(self,  args: TrainingArguments, state: TrainerState, control, **kwargs):
        print("Starting training")

    # def on_train_begin(self, args: TrainingArguments, state :  TrainerState, control, **kwargs):
    #     """
    #     Event called at the end of training.
    #     """
    #     print(state)

    #     try:
    #        exp_log = self.experiment.log("train_end_parameters", state, LogType.Table)
    #        print(exp_log)
    #     except Exception as e:
    #         print("can't send log")

    def on_log(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        """
        Event called after logging the last logs.
        """
        # print(state.log_history)


        # Keep track of train and evaluate loss.
        loss_history = {'train_loss':[], 'train_epochs':[]}
        # Loop through each log history.
        for log_history in state.log_history:

            if 'loss' in log_history.keys():
            # Deal with trianing loss.
                print('epoch:', log_history['epoch'])
                print('train_loss:', log_history['loss'])
                loss_history['train_loss'].append(log_history['loss'])
                loss_history['train_epochs'].append(log_history['epoch'])

        print(loss_history)

        # self._log_metric("training_log_history", loss_history,  LogType.LINE)


    # def _log_metric(self, name: str, value: float, retry: int):
    #     try:
    #         self.experiment.log(name=name, data=value, type=LogType.LINE, )
    #     except Exception:
    #         logging.exception(f"couldn't log {name}")
    #         if retry > 0:
    #             logging.info(f"retrying log {name}")
    #             self._log_metric(name, value, retry-1)



