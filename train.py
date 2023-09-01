# Import required libraries
import os
import albumentations

from picsellia import Client
from picsellia.types.enums import InferenceType, LogType
from transformers import Trainer, TrainingArguments
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import detr
import processor_picsellia_utils

project_name="Sample-project"
experiment_name="DETR_huggingface_transformer"
train_version_name="training"
eval_version_name="eval"
train_set_local_dir="train_dataset_version"
eval_set_local_dir="eval_dataset_version"
finetuned_output_dir="finetuned_detr"
coco_annotations_path = "train_annotations.json"
cwd = os.getcwd()


# Initializing Picsellia connection
client = Client(api_token="", organization_name="Nwoke", host="https://trial.picsellia.com")

# Retrieve the experiment
project = client.get_project(project_name)
experiment = project.get_experiment(experiment_name)
attached_datasets = experiment.list_attached_dataset_versions()
train_dataset_version = experiment.get_dataset(train_version_name)
eval_dataset_version = experiment.get_dataset(eval_version_name)

images_dir = os.path.join(os.getcwd(), train_set_local_dir)

labels = train_dataset_version.list_labels()
label_names = [label.name for label in labels]
labelmap = {str(i): label.name for i, label in enumerate(labels)}


if os.path.isdir(train_set_local_dir):
  print("dataset version has already been downloaded")
else:
  train_dataset_version.download(train_set_local_dir)


if os.path.isdir(eval_set_local_dir):
  print("dataset version has already been downloaded")
else:
  eval_dataset_version.download(eval_set_local_dir)



# Prepare Data in COCO Format
annotations_coco = processor_picsellia_utils.coco_format_loader(train_dataset_version, label_names, coco_annotations_path)
cats = annotations_coco.cats
id2label = {str(k): v['name'] for k,v in cats.items()}
label2id = {v: k for k, v in id2label.items()}

# log dataset labels to the picsellia experiment
experiment.log("labelmap", id2label, type= LogType.TABLE, replace=True)


# Load transformer image processor for DetrModel
checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# Load albumentations library for image augumentation
transform_ = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

# Preprocess, Augument and Transform Data
train_dataset = processor_picsellia_utils.CocoDetection(img_folder=images_dir,
                                        ann_file = coco_annotations_path,
                                        processor=image_processor,
                                        transform=transform_)

print("Number of training examples:", len(train_dataset))

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


# Training the DETR model
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


training_args = TrainingArguments(
    output_dir="detr-resnet-50_finetuned_cppe5",
    # overwrite_output_dir= ,
    per_device_train_batch_size=8,
    num_train_epochs=40,
    fp16=False,
    save_steps=200,
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Log hyperparameters to Picesllia
training_hyp_params  = training_args.to_dict()
experiment.log("hyper-parameters", training_hyp_params, type=LogType.TABLE)

# initialize trainer callback
picsellia_callback = detr.CustomPicselliaCallback(experiment=experiment)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    tokenizer=image_processor,
    callbacks=[picsellia_callback],
)


trainer.train()
trainer.save_model(finetuned_output_dir)
finetuned_model_path = os.path.join(os.getcwd(), finetuned_output_dir)

processor_picsellia_utils.finetuned_model_to_picsellia(experiment, cwd, save_dir=finetuned_output_dir)

detr.picsellia_detr_evaluator(experiment, eval_set_local_dir, eval_version_name, label_names, finetuned_output_dir)
experiment.compute_evaluations_metrics(inference_type=InferenceType.OBJECT_DETECTION)
