import os


def setup_hyp(
    experiment=None,
    data_json_path=None,
    params={},
    cwd=None,
    task="detect",
):

    opt = Opt()

    opt.task = task
    opt.mode = "train"
    opt.cwd = cwd
    
    # Train settings -------------------------------------------------------------------------------------------------------
    # opt.model = weight_path
    opt.data = data_json_path
    opt.epochs = 100 if not "num_train_epochs" in params.keys() else params["num_train_epochs"]
    opt.batch = 8 if not "per_device_train_batch_size" in params.keys() else params["per_device_train_batch_size"]
    # opt.imgsz = 640 if not "input_shape" in params.keys() else params["input_shape"]
    opt.save = True
    # opt.save_period = (
    #     100 if not "save_period" in params.keys() else params["save_period"]
    # )
    opt.fp16 = True if not "fp16" in params.keys() else params["fp16"]
    opt.cache = False
    opt.device = "0" if "gpu" else "cpu"
    opt.project = cwd
    opt.name = "exp"
    opt.exist_ok = False
    opt.pretrained = True
    opt.optimizer = "Adam"
    opt.logging_steps = 50
    opt.logging_dir = cwd if not "logging_dir" in params.keys() else params["logging_dir"]

    # Hyperparameters ------------------------------------------------------------------------------------------------------
    opt.lr0 = 0.01  if not "learning_rate" in params.keys() else params["learning_rate"] # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    # opt.lrf = 0.01  # final learning rate (lr0 * lrf)
    opt.momentum = 0.937  # SGD momentum/Adam beta1
    opt.weight_decay = 0.0005  if not "weight_decay" in params.keys() else params["weight_decay"]  # optimizer weight decay 5e-4
    opt.lr_scheduler_type = "linear" if not "lr_scheduler_type" in params.keys() else params["lr_scheduler_type"]
    opt.warmup_steps = 0.0  # warmup steop

    return opt

class Opt:
    pass



# def send_run_to_picsellia(experiment, cwd, save_dir=None, imgsz=640):
#     if save_dir is not None:
#         final_run_path=save_dir
#     else:
#         final_run_path = find_final_run(cwd)
#     best_weigths, hyp_yaml = get_weights_and_config(final_run_path)

#     model_latest_path = os.path.join(final_run_path, 'weights', 'best.onnx')
#     model_dir = os.path.join(final_run_path, 'weights')
#     if os.path.isfile(os.path.join(model_dir, "best.onnx")):
#         model_latest_path = os.path.join(model_dir, "best.onnx")
#     elif os.path.isfile(os.path.join(model_dir, "last.onnx")):
#         model_latest_path = os.path.join(model_dir, "last.onnx")
#     elif os.path.isfile(os.path.join(model_dir, "best.pt")):
#         checkpoint_path = os.path.join(model_dir, 'best.pt')
#         model = YOLO(checkpoint_path)
#         model.export(format='onnx', imgsz=imgsz, task='detect')
#         model_latest_path = os.path.join(final_run_path, 'weights', 'best.onnx')
#     elif not os.path.isfile(os.path.join(model_dir, "last.pt")):
#         checkpoint_path = os.path.join(model_dir, 'last.pt')
#         model = YOLO(checkpoint_path)
#         model.export(format='onnx', imgsz=imgsz, task='detect')
#         model_latest_path = os.path.join(final_run_path, 'weights', 'last.onnx')
#     else:
#         logging.warning("Can't find last checkpoints to be uploaded")
#         model_latest_path = None
#     if model_latest_path is not None:
#         experiment.store('model-latest', model_latest_path)
#     if best_weigths is not None:
#         experiment.store('checkpoint-index-latest', best_weigths)
#     if hyp_yaml is not None:
#         experiment.store('checkpoint-data-latest', hyp_yaml)
#     for curve in get_metrics_curves(final_run_path):
#         if curve is not None:
#             name = curve.split('/')[-1].split('.')[0]
#             experiment.log(name, curve, LogType.IMAGE)
#     for batch in get_batch_mosaics(final_run_path):
#         if batch is not None:
#             name = batch.split('/')[-1].split('.')[0]
#             experiment.log(name, batch, LogType.IMAGE)
