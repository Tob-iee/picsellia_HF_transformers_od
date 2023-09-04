import os
import logging

import torch
from PIL import Image

from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection
from transformers import TrainerCallback, TrainingArguments, TrainerState

from picsellia import Experiment
from picsellia.types.enums import LogType


class CustomPicselliaCallback(TrainerCallback):

    def __init__(self, experiment: Experiment):
        self.experiment=experiment


    def on_train_begin(self,  args: TrainingArguments, state: TrainerState, control, **kwargs):
        print("Starting training")


    def on_train_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        """
        Event called at the end of training.
        """
        # Keep track of train and evaluate loss.'
        print("state.log_history:", state.log_history)

        # Loop through each log history.
        for log_history in state.log_history:
            if 'loss' in log_history.keys():

                # Deal with trianing loss.
                loss = log_history['loss']
                learning_rate_decay = log_history['learning_rate']
                print('train_loss:', loss)
                print('train_lr-decay:', learning_rate_decay)

                try:
                    self._log_metric("loss_training_hist", loss,  LogType.LINE)
                    self._log_metric("lr-decay_hist", learning_rate_decay,  LogType.LINE)
                except Exception as e:
                    print("can't send log")


    def _log_metric(self, name: str, value: float, retry: int):
        try:
            self.experiment.log(name=name, data=value, type=LogType.LINE, replace=True)
        except Exception:
            logging.exception(f"couldn't log {name}")
            if retry > 0:
                logging.info(f"retrying log {name}")
                self._log_metric(name, value, retry-1)


def picsellia_detr_evaluator(experiment, eval_set_local_dir, eval_version_name, label_names, finetuned_output_dir):
    image_processor = AutoImageProcessor.from_pretrained(finetuned_output_dir)
    model = AutoModelForObjectDetection.from_pretrained(finetuned_output_dir)

    picsellia_eval_ds = experiment.get_dataset(name=eval_version_name)
    labels_picsellia = {k: picsellia_eval_ds.get_label(k) for k in label_names}

    eval_image_path = os.path.join(os.getcwd(), eval_set_local_dir)
    list_eval_image = os.listdir(eval_image_path)

    for img_path in list_eval_image:
        eval_image_ = os.path.join(eval_image_path, img_path)
        print(eval_image_)
        eval_image = Image.open(eval_image_)


        with torch.no_grad():
            inputs = image_processor(images=eval_image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([eval_image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

        rectangles = [] # ((x, y, w, h), label, confidence)


        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            fname = eval_image_.split('/')[-1]
            asset = picsellia_eval_ds.find_asset(filename=fname)
            box = [int(i) for i in box.tolist()]
            x, y, x1, y1 = box
            label_name_detected = model.config.id2label[label.item()]
            picsellia_label_object = labels_picsellia[label_name_detected] # label (picsellia one)
            conf = float(score)
            rectangles.append((x, y, x1 - x, y1 - y, picsellia_label_object, conf))
            experiment.add_evaluation(asset, rectangles=rectangles)


            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

    return "Evaluation Done"