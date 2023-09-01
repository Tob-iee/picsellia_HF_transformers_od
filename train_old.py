# Import required libraries
import os
import io

import numpy as np
from PIL import Image

import albumentations
from pycocotools.coco import COCO

from picsellia import Client
from picsellia import Experiment
from picsellia.types.enums import InferenceType, LogType
from picsellia.exceptions import ResourceNotFoundError

import torchvision
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers import AutoImageProcessor
from transformers import AutoModelForObjectDetection

from processor_picsellia_utils import processing, detection_trainer


# Initializing Picsellia connection
client = Client(api_token="329433da34f56dff854da2ac8795c918f710c15f", organization_name="Nwoke", host="https://trial.picsellia.com")

# Retrieve the experiment
project = client.get_project("Sample-project")
experiment = project.get_experiment("train_dataV_experiment")
attached_datasets = experiment.list_attached_dataset_versions()

current_dir = os.path.join(os.getcwd(), experiment.base_dir)
base_imgdir = experiment.png_dir

dataset =  client.get_dataset_by_id('0189b100-2131-772b-921d-b83210541cf7')
dataset_version = client.get_dataset_version_by_id('0189b100-4d4b-7e81-b729-2978c654d00d')

# dataset_version.download('dataset_train_version')
labels = dataset_version .list_labels()
label_names = [label.name for label in labels]
labelmap = {str(i): label.name for i, label in enumerate(labels)}



for data_type, dataset in {"train": dataset_version}.items():
    print(data_type, dataset)
coco_annotations_path = "0189b100-4d4b-7e81-b729-2978c654d00d_annotations.json"
annotations_coco = COCO(coco_annotations_path)

cats = annotations_coco.cats
id2label = {str(k): v['name'] for k,v in cats.items()}

label2id = {v: k for k, v in id2label.items()}


dataset_path =  os.path.join(base_imgdir, data_type)
images_dir = os.path.join(dataset_path, 'images')



dataset_processed = []
annotations = []
images_json = []

for key in annotations_coco.anns.keys():
    annotations.append(annotations_coco.anns[key])


for key in annotations_coco.imgs.keys():
    images_json.append(annotations_coco.imgs[key])

for images in images_json:
    filename = images["file_name"]
    image_path = os.path.join(images_dir, filename)
    image = Image.open(image_path)

    # # Encode your PIL Image as a JPEG without writing to disk
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=75)

    ids_list, area_list, bboxes_list, categories_list = [], [], [], []

    for ann in annotations:
        if images["id"] == int(ann["image_id"]):
            ids_list.append(ann["id"])
            area_list.append(ann["area"])
            bboxes_list.append(ann["bbox"])
            categories_list.append(ann["category_id"])

    dict_of_data = {'image_id': images["id"],
                    'image': image,
                    'objects': {'id': ids_list,
                        'area': area_list,
                        'bbox': bboxes_list,
                        'category': categories_list}}

    dataset_processed.append(dict_of_data)

dataset_processed.pop(23)

# Preprocess, Augument and Transform Data
# Load transformer image processor for DetrModel
checkpoint =  "_detr-resnet-50_finetuned_cppe5/original_checkpoint"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# image_processor.save_pretrained("_detr-resnet-50_finetuned_cppe5/image_processor")


# Load albumentations library for image augumentation
transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)


# Reformats each image's annotations for to the image_processor
def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


# Transforming a batch of images
# It auguments images using albumentations and preprocess images with image_processor
def transform_aug_ann(example):
    ids_, images, bboxes, area, categories = [], [], [], [], []

    # for example in examples:
    image_ids = example["image_id"]
    if image_ids != 23:
        objects =  example["objects"]
        image = example["image"]

        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        ids_.append(image_ids)
        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(ids_, categories, area, bboxes)
    ]

    encoding = image_processor(images=images, annotations=targets, return_tensors="pt")
    pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
    target = encoding["labels"][0] # remove batch dimension

    return pixel_values, target


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

transformed_list = []

for processed in dataset_processed:
    # print(processed)
    tranformed = transform_aug_ann(processed)
    transformed_list.append(tranformed)

experiment.log("labelmap", id2label, type= LogType.TABLE, replace=True)

cwd = os.getcwd()

# Training the DETR model
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# model.save_pretrained("_detr-resnet-50_finetuned_cppe5/checkpoint")


training_args = TrainingArguments(
    output_dir="detr-resnet-50_finetuned_cppe5",
    # overwrite_output_dir= ,
    per_device_train_batch_size=8,
    num_train_epochs=30,
    fp16=False,
    save_steps=200,
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=1e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Log hyperparameters to Picesllia
training_hyp_params  = training_args.to_dict()
experiment.log("hyper-parameters", training_hyp_params, type=LogType.TABLE)

picsellia_callback = detection_trainer.CustomPicselliaCallback(experiment=experiment)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=transformed_list,
    tokenizer=image_processor,
    callbacks=[picsellia_callback],
)

# trainer.add_callback(picsellia_callback(trainer))

trainer.train()

save_dir = "detr-resnet-50_finetuned_cppe5_n"

trainer.save_model(save_dir)
processing.send_run_to_picsellia(experiment, cwd, save_dir=save_dir)


picsellia_detr_evaluator(experiment, eval_set_local_dir, eval_version_name, label_names, finetuned_output_dir)
experiment.compute_evaluations_metrics(inference_type=InferenceType.OBJECT_DETECTION)
