import os
import pathlib
import argparse
import model_config as mc


"""
call the training script which is in 
models/research/object_detection/model_main_tf2.py
"""

if __name__ == '__main__':

    models_repo_root = mc.TF_MODEL_REPO_PATH
    abs_models_repo_path = pathlib.Path(models_repo_root).absolute()

    research_path = f"{abs_models_repo_path}/models/research"
    object_detection_path = f"{research_path}/object_detection"

    os.system(f"python {object_detection_path}/model_main_tf2.py \
    --pipeline_config_path {mc.CHECKPOINT_PATH}/{mc.base_pipeline_file} \
    --model_dir={mc.CHECKPOINT_PATH} \
    --alsologtostderr \
    --num_train_steps={mc.num_training_steps}"
    )



