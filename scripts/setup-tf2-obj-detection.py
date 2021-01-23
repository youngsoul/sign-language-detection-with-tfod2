import os
import subprocess
import pathlib
import platform
import logging
import model_config as mc
import tarfile
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

base_tf2_pretrained_weights_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711"

this_file_dir = os.path.dirname(os.path.abspath(__file__))


def clone_models_repo(root_dir:str):
    # if the models directory already exists DO NOT clone the repo again
    if not pathlib.Path(f"{root_dir}/models").exists():
        os.chdir(root_dir)
        os.system("git clone https://github.com/tensorflow/models.git")

def install_tf_detection_api(research_dir:str):
    os.chdir(research_dir)
    logger.info(f"CWD: {research_dir}")
    logger.info("Running protoc")
    os.system("protoc object_detection/protos/*.proto --python_out=.")
    os.system("cp object_detection/packages/tf2/setup.py .")
    logger.info("pip install Tensorflow Dependendencies. see setup.py")
    os.system("pip install .")

def verify_tf_obj_detection_install(research_dir:str):
    os.chdir(research_dir)
    os.system("python object_detection/builders/model_builder_tf2_test.py")

def fix_tf_utils_bug():
    print(this_file_dir)
    if pathlib.Path(f"{this_file_dir}/tf_util_fixed.txt").exists():
        logger.info("tf_util.py already updated....")
        return

    logger.warning(f"NOTE:  Updating tf_utils.py")
    # ./python3.8/site-packages/tensorflow/python/keras/utils/tf_utils.py
    venv_python_path = subprocess.check_output("which python", shell=True)
    x = str(venv_python_path).split("/")[:-2]
    p = "/".join(x[1:])
    py_version_tuple = platform.python_version_tuple()
    path_to_tf_utls = "/" + p + f"/lib/python{py_version_tuple[0]}.{py_version_tuple[1]}/site-packages/tensorflow/python/keras/utils/tf_utils.py"

    print(path_to_tf_utls)

    with open(path_to_tf_utls) as f:
        tf_utils = f.read()

    with open(path_to_tf_utls, 'w') as f:
        throw_statement = "raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))"
        tf_utils = tf_utils.replace(throw_statement, "if not isinstance(x, str):" + throw_statement)
        f.write(tf_utils)

    os.chdir(this_file_dir)
    os.system(f"touch {this_file_dir}/tf_util_fixed.txt")

def pip_install_labelimg():
    os.system("pip install labelImg")


def download_pretrained_model_weights():
    deploy_dir = pathlib.Path(mc.PRETRAINED_MODEL_PATH)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    if pathlib.Path(f"{mc.PRETRAINED_MODEL_PATH}/{mc.pretrained_checkpoint}").exists():
        # then do not download it again
        logger.info(f"{mc.pretrained_checkpoint} file already exists.")
        logger.info(f"To download again, delete: {mc.PRETRAINED_MODEL_PATH}/{mc.pretrained_checkpoint}")
        return

    os.chdir(str(deploy_dir))
    logger.info(f"Downloading pretrained weights to: {str(deploy_dir)}")

    download_tar_url = f"{base_tf2_pretrained_weights_url}/{mc.pretrained_checkpoint}"
    logger.info(f"execute: wget {download_tar_url}")
    os.system(f"wget {download_tar_url}")

    tar = tarfile.open(f"{str(deploy_dir)}/{mc.pretrained_checkpoint}")
    tar.extractall()
    tar.close()

    pathlib.Path(f"{str(deploy_dir)}/{mc.pretrained_checkpoint}").unlink()



def copy_base_training_config():
    # models/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config

    source_dir = pathlib.Path(f"{mc.PRETRAINED_MODEL_PATH}/{mc.model_name}")

    os.chdir(mc.PRETRAINED_MODEL_PATH)
    source_base_config_path = f"{source_dir}/{mc.base_pipeline_file}"
    pipeline_fname = f"{mc.CHECKPOINT_PATH}/{mc.base_pipeline_file}"

    pathlib.Path(mc.CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    if pathlib.Path(pipeline_fname).exists():
        logger.info(f"WARNING: Model Config: {pipeline_fname} already exists")
        logger.info(f"Config file not overwritten")
        return

    logger.info(f"cp {source_base_config_path} {mc.CHECKPOINT_PATH}")
    os.system(f"cp {source_base_config_path} {mc.CHECKPOINT_PATH}")

    return

def create_label_map_file():
    tf_annotations_dir = mc.TF_ANNOTATION_PATH
    """
    item {
    id: 1
    name: 'Raspberry_Pi_3'
}
item {
    id: 2
    name: 'Arduino_Nano'
}
item {
    id: 3
    name: 'ESP8266'
}
item {
    id: 4
    name: 'Heltec_ESP32_Lora'
}
    :return:
    :rtype:
    """

    with open(f"{tf_annotations_dir}/{mc.label_map_fname}", "w" ) as f2:
        for i, line in enumerate(mc.classes):
            f2.write("item {\n")
            f2.write(f"\tid: {i+1}\n")
            f2.write(f"\tname: '{line.strip()}'\n")
            f2.write("}\n")


def update_model_config_file():

    config_file_path = f"{mc.CHECKPOINT_PATH}/{mc.base_pipeline_file}"

    config = config_util.get_configs_from_pipeline_file(config_file_path)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_file_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(mc.classes)
    pipeline_config.train_config.batch_size = mc.batch_size
    pipeline_config.train_config.fine_tune_checkpoint = mc.PRETRAINED_MODEL_PATH + f'/{mc.model_name}/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = mc.TF_ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [mc.TF_ANNOTATION_PATH + '/train.record']
    pipeline_config.eval_input_reader[0].label_map_path = mc.TF_ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [mc.TF_ANNOTATION_PATH + '/test.record']

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(config_file_path, "wb") as f:
        f.write(config_text)

if __name__ == '__main__':
    models_repo_root = mc.TF_MODEL_REPO_PATH

    abs_models_repo_path = pathlib.Path(models_repo_root).absolute()
    abs_models_repo_path.mkdir(parents=True, exist_ok=True)
    research_path = f"{abs_models_repo_path}/models/research"
    object_detection_path = f"{research_path}/object_detection"

    print(abs_models_repo_path)
    print(research_path)

    # 1 Clone Tensorflow Repos Dir
    logger.info(f"Cloning Tensorflow Models Repo")
    clone_models_repo(root_dir=models_repo_root)
    logger.info("Done")

    # 2 Install the Object Detection API
    logger.info("Installing Tensorflow Detection API...")
    install_tf_detection_api(research_dir=research_path)
    logger.info("Done")


    # 3 Verify TF ObjDet Install
    logger.info("Verify Tensorflow Object Detection Install...")
    verify_tf_obj_detection_install(research_dir=research_path)
    logger.info("Done")

    # 4 Fix tf_util bug
    logger.info("Fix tf_utils bug.  Watch for this step to no longer be needed")
    fix_tf_utils_bug()
    logger.info("Done")

    # 5 install LabelImg
    logger.info("Install LabelImg")
    pip_install_labelimg()

    # 6 download pretrained weights for selected model
    logger.info(f"Download pretrained model weights for model: {mc.pretrained_checkpoint}")
    download_pretrained_model_weights()

    # 7 copy base config for model
    logger.info(f"Copy base model configuration: {mc.base_pipeline_file}")
    copy_base_training_config()

    # 8 create label_map.pbtxt file, if it not not already there
    create_label_map_file()

    # 9 update model configuration file
    update_model_config_file()