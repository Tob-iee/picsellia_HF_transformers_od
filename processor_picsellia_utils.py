import os
import json
import numpy as np
import torchvision
from pycocotools.coco import COCO
from picsellia import Client
from picsellia.types.enums import InferenceType, Framework


def coco_format_loader(train_dataset_version, label_names, coco_annotations_path):
    coco_annotation = train_dataset_version.build_coco_file_locally(
            enforced_ordered_categories=label_names
        )

    annotations_dict = coco_annotation.dict()

    with open(coco_annotations_path, "w") as f:
        f.write(json.dumps(annotations_dict))

    annotations_coco = COCO(coco_annotations_path)
    return annotations_coco


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder,ann_file, processor, transform, train=True):
        # ann_file = "0189b100-4d4b-7e81-b729-2978c654d00d_annotations.json"
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor
        self.transform = transform

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        ids_list, area_list, bboxes_list, categories_list = [], [], [], []

        for objects in target:
            ids_list.append(objects["id"])
            area_list.append(objects["area"])
            bboxes_list.append(objects["bbox"])
            categories_list.append(objects["category_id"])

        dict_of_data = {'image_id': self.ids[idx],
                        'image': img,
                        'objects': {'id': ids_list,
                            'areas': area_list,
                            'bboxes': bboxes_list,
                            'category_ids': categories_list}}


        # Transforming a batch of images
        # It auguments images using albumentations and preprocess images with image_processor
        ids_trans, images_trans, bboxes_trans, area_trans, categories_trans = [], [], [], [], []

        image_ids = dict_of_data["image_id"]

        objects =  dict_of_data["objects"]
        image = dict_of_data["image"]

        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = self.transform(image=image, bboxes=objects["bboxes"], category=objects["category_ids"])

        ids_trans.append(objects["id"])
        area_trans.append(objects["areas"])
        images_trans.append(out["image"])
        bboxes_trans.append(out["bboxes"])
        categories_trans.append(out["category"])

        # Reformats each image's annotations(targets) for the image_processor
        targets = [
                  {"image_id": image_ids, "annotations": self.formatted_anns(ids_, cat_, ar_, box_)}
                  for ids_, cat_, ar_, box_ in zip(ids_trans, categories_trans, area_trans, bboxes_trans)
                  ]

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.processor(images=images_trans, annotations=targets, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target_lab = encoding["labels"][0] # remove batch dimension

        return pixel_values, target_lab


    def formatted_anns(self, image_id, category, area, bbox):
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id[i],
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)

        return annotations


def finetuned_model_to_picsellia(experiment, cwd, save_dir=None):
    # send the trained model to picsellia
    model_latest_path = os.path.join(cwd, save_dir)
    file_list = os.listdir(model_latest_path)
    print(file_list)

    for files in file_list:
        model_latest_files = os.path.join(model_latest_path, files)
        experiment.store(files, model_latest_files)


def save_model_version(experiment):
    model_name = "detr-resnet-50_"
    model_description = "Finetuned DETR model"

    if 'api_token' not in os.environ:
        raise Exception("You must set an api_token to run this image")
    api_token = os.environ["api_token"]

    if "host" not in os.environ:
        host = "https://trial.picsellia.com"
    else:
        host = os.environ["host"]

    if "organization_name " not in os.environ:
        organization_name = None
    else:
        organization_name = os.environ["organization_name"]

    client = Client(
        api_token=api_token,
        host=host,
        organization_name=organization_name
    )

    try:
        my_model = client.create_model(
        name=model_name,
        description=model_description,
        )
    except:
         # Get the model that will receive this new version
        my_model = client.get_model(
        name=model_name,
        )

    # Get the model that will receive this new version
    my_model = client.get_model(
    name=model_name,
    )


    experiment.export_in_existing_model(my_model)

    # # Define the model version Inference type
    # inference_type = InferenceType.OBJECT_DETECTION

    # # Define the model version Framework
    # framework = Framework.PYTORCH

    # model_version = my_model.create_version(
    # name=f'{model_name}v{model_version_no}', # <- The name of your ModelVersion
    # labels=labelmap,
    # type=inference_type,
    # framework=framework
    # )

    # model_version = my_model.get_version(f'{model_name}v{str(model_version_no)}')
    # model_version.store(f"model-version_{model_version_no}", fine_tuned_model_path, do_zip=True)

def get_experiment():
    if 'api_token' not in os.environ:
        raise Exception("You must set an api_token to run this image")
    api_token = os.environ["api_token"]

    if "host" not in os.environ:
        host = "https://trial.picsellia.com"
    else:
        host = os.environ["host"]

    if "organization_name " not in os.environ:
        organization_name = None
    else:
        organization_name = os.environ["organization_name"]

    client = Client(
        api_token=api_token,
        host=host,
        organization_name=organization_name
    )

    # Retrieve experiment & dataset versions
    if "experiment_name" in os.environ:
        experiment_name = os.environ["experiment_name"]
        if "project_token" in os.environ:
            project_token = os.environ["project_token"]
            project = client.get_project_by_id(project_token)
        elif "project_name" in os.environ:
            project_name = os.environ["project_name"]
            project = client.get_project(project_name)
        experiment = project.get_experiment(experiment_name)
    else:
        raise Exception("You must set the project_token or project_name and experiment_name")
    return experiment