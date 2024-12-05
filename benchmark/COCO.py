import torch
import torch.nn as nn
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
import yaml
import os
import argparse

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_from_yaml(yaml_path):
    # Load YAML configuration
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Import and initialize model dynamically
    module = __import__(config["model"]["module"], fromlist=[config["model"]["class"]])
    model_class = getattr(module, config["model"]["class"])
    model = model_class(**config["model"]["params"])
    
    return model, config


def main(args):
    # Step 1: Load Model and Configuration
    yaml_path = args.yaml
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = "cpu"

    model, config = load_model_from_yaml(yaml_path)

    
    # Print Model Details
    print(f"Loaded Model: {config['model']['class']}")
    print(f"Number of Parameters: {count_parameters(model)}")

    # Step 2: Detectron2 Configuration
    cfg = get_cfg()
    cfg.merge_from_file(config["detectron2"]["config_file"])
    cfg.DATASETS.TRAIN = tuple(config["detectron2"]["datasets"]["train"])
    cfg.DATASETS.TEST = tuple(config["detectron2"]["datasets"]["test"])
    cfg.DATALOADER.NUM_WORKERS = config["detectron2"]["dataloader"]["num_workers"]
    cfg.SOLVER.IMS_PER_BATCH = config["detectron2"]["solver"]["ims_per_batch"]
    cfg.SOLVER.BASE_LR = config["detectron2"]["solver"]["base_lr"]
    cfg.SOLVER.MAX_ITER = config["detectron2"]["solver"]["max_iter"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config["detectron2"]["model"]["num_classes"]
    cfg.MODEL.WEIGHTS = ""  # Train from scratch
    cfg.OUTPUT_DIR = config["detectron2"]["output_dir"]
    cfg.MODEL.DEVICE = device
    # Create output directory if not exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Step 3: Training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Step 4: Evaluation
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    results = inference_on_dataset(trainer.model, val_loader, evaluator)

    # Print Results
    print("Evaluation Results:", results)

if __name__ == "__main__":
    """
    Here is a launch example:
    CUDA_VISIBLE_DEVICES=2 python COCO.py --yaml /home/jiahuan/workspace/Toynetwork/cfg/coco_transformer.yaml --device cuda
    """
    parser = argparse.ArgumentParser(description="Train and Evaluate Detectron2 Model")
    parser.add_argument("--yaml", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()
    main(args)