from deep_privacy import cli
import os
import re
import pathlib
import typing
from deep_privacy import logger
from deep_privacy.inference.deep_privacy_anonymizer import DeepPrivacyAnonymizer
from deep_privacy.build import available_models
import torch
from deep_privacy.config import Config
from deep_privacy import logger, torch_utils
from deep_privacy.inference.infer import load_model_from_checkpoint

video_suffix = [".mp4"]
image_suffix = [".jpg", ".jpeg", ".png"]

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_source_files(source_path: str):
    source_path = pathlib.Path(source_path)
    assert source_path.is_file() or source_path.is_dir(),\
        f"Did not find file or directory: {source_path}"
    if source_path.is_file():
        return [source_path]
    relevant_suffixes = image_suffix + video_suffix
    file_paths = recursive_find_file(source_path, relevant_suffixes)
    return file_paths

def recursive_find_file(folder: pathlib.Path,
                        suffixes: typing.List[str]
                        ) -> typing.List[pathlib.Path]:
    files = []
    for child in folder.iterdir():
        if not child.is_file():
            child_files = recursive_find_file(child, suffixes)
            files.extend(child_files)
        if child.suffix in suffixes:
            files.append(child)
    return files


def get_target_paths(source_paths: typing.List[pathlib.Path],
                     target_path: str,
                     default_dir: pathlib.Path):
    if not target_path is None:
        target_path = pathlib.Path(target_path)
        if len(source_paths) > 1:
            target_path.mkdir(exist_ok=True, parents=True)
            target_paths = []
            for source_path in source_paths:
                target_paths.append(target_path.joinpath(source_path.name))
            return target_paths
        else:
            target_path.parent.mkdir(exist_ok=True)
            return [target_path]
    logger.info(
        f"Found no target path. Setting to default output path: {default_dir}")
    default_target_dir = default_dir
    target_path = default_target_dir
    target_path.mkdir(exist_ok=True, parents=True)
    target_paths = []
    for source_path in source_paths:
        if source_path.suffix in video_suffix:
            target_path = default_target_dir.joinpath("anonymized_videos")
        else:
            target_path = default_target_dir.joinpath("anonymized_images")
        target_path = target_path.joinpath(source_path.name)
        os.makedirs(target_path.parent, exist_ok=True)
        target_paths.append(target_path)
    return target_paths

def build_anonymizer(
        model_name=available_models[0],
        batch_size: int = 1,
        fp16_inference: bool = True,
        truncation_level: float = 0,
        detection_threshold: float = .1,
        opts: str = None,
        config_path: str = None,
        return_cfg=False) -> DeepPrivacyAnonymizer:
    """
        Builds anonymizer with detector and generator from checkpoints.

        Args:
            config_path: If not None, will override model_name
            opts: if not None, can override default settings. For example:
                opts="anonymizer.truncation_level=5, anonymizer.batch_size=32"
    """
    print("manual config override: ", config_path)
    if config_path is None:
        print(config_path)
        assert model_name in available_models,\
            f"{model_name} not in available models: {available_models}"
        #cfg = get_config(config_urls[model_name])
    else:
        cfg = Config.fromfile(config_path)
    logger.info("Loaded model:" + cfg.model_name)
    generator = load_model_from_checkpoint(cfg)
    logger.info(f"Generator initialized with {torch_utils.number_of_parameters(generator)/1e6:.2f}M parameters")
    cfg.anonymizer.truncation_level = truncation_level
    cfg.anonymizer.batch_size = batch_size
    cfg.anonymizer.fp16_inference = fp16_inference
    cfg.anonymizer.detector_cfg.face_detector_cfg.confidence_threshold = detection_threshold
    cfg.merge_from_str(opts)
    anonymizer = DeepPrivacyAnonymizer(generator, cfg=cfg, **cfg.anonymizer)
    if return_cfg:
        return anonymizer, cfg
    return anonymizer


def anonymize_images(input_folder = 'Images\\CameraTest', output_folder = 'Images\\Anonymized'):

    # get current directory
    path = os.getcwd()
    print("Current Directory", path)
    print("torch? ", torch.cuda.is_available())
         
    anonymizer, cfg = build_anonymizer(
    model_name='deep_privacy_V1', opts=None, config_path= r'C:\Users\Chris\Desktop\PhDProject\GaitMonitor\Code\Programs\Data_Processing\Model_Based\DeepPrivacy\configs\fdf/deep_privacy_v1.py',
    return_cfg=True)

    #Cycle through all images and subdirectories and do it manually
    for (subdir, dirs, files) in os.walk(input_folder):
        dirs.sort(key=numericalSort)
        for file_iter, file in enumerate(sorted(files, key = numericalSort)):
            tmp = [input_folder, subdir, file]
            output_subfolder = "\\".join(subdir.split('\\')[-2:])
            print("tmps:", tmp)
            print("OUTPUT SUBFOLDER IS: ", output_subfolder)
            if os.path.exists(output_folder + "\\" + output_subfolder) == False:
                print("Trying to make: ", output_folder + "\\" + output_subfolder)
                os.makedirs(output_folder + "\\" + output_subfolder, exist_ok=True)
            file_destination = output_folder + "\\" + output_subfolder + "\\" + file
            print("processing file: ", os.path.join(*tmp), "outputting to : ", file_destination)

            #cli.main(os.path.join(*tmp), file_destination)
            
            #output_dir = cfg.output_dir
            #source_paths = get_source_files(input_folder)
            #image_paths = [source_path for source_path in source_paths
            #            if source_path.suffix in image_suffix]

            #image_target_paths = []
            #if len(image_paths) > 0:
            #    image_target_paths = get_target_paths(
            #        image_paths, output_folder,
            #        output_dir)
            #assert len(image_paths) == len(image_target_paths)

            #if len(image_paths) > 0:
            anonymizer.anonymize_image_paths([pathlib.Path(os.path.join(*tmp))], [pathlib.Path(file_destination)])

if __name__ == "__main__":
    print("this is main")
    anonymize_images(input_folder=r'C:\Users\Chris\Desktop\PhDProject\GaitMonitor\Code\Datasets\WeightGait\Full_Dataset',
                      output_folder=r'C:\Users\Chris\Desktop\PhDProject\GaitMonitor\Code\Datasets\WeightGait\Anonymized')
