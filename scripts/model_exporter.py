import os
import pathlib
import model_config as mc


"""
ASSUMES you are in the object_detection directory
python exporter_main_v2.py \
    --trained_checkpoint_dir training \
    --output_directory inference_graph \
    --pipeline_config_path training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config
"""

if __name__ == '__main__':
    models_repo_root = mc.TF_MODEL_REPO_PATH
    abs_models_repo_path = pathlib.Path(models_repo_root).absolute()

    object_detection_path = f"{abs_models_repo_path}/models/research/object_detection"

    pathlib.Path(f"{mc.EXPORTED_MODEL_PATH}").mkdir(parents=True, exist_ok=True)

    os.system(f"python {object_detection_path}/exporter_main_v2.py \
    --trained_checkpoint_dir {mc.CHECKPOINT_PATH} \
    --output_directory {mc.EXPORTED_MODEL_PATH} \
    --pipeline_config_path {mc.CHECKPOINT_PATH}/{mc.base_pipeline_file}")



