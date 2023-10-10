from pathlib import Path
import argparse
import numpy as np
import cv2
import json
import pycocotools.mask as mask_util
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import BoxMode
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger

logger = setup_logger()


def export_json(instances: Instances, img_id: int, output_file: Path):
    json_data = []

    if instances.has("pred_boxes"):
        boxes = BoxMode.convert(
            instances.pred_boxes.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
        )

    if instances.has("pred_masks"):
        rles = [
            mask_util.encode(
                np.array(mask.cpu()[:, :, None], order="F", dtype="uint8")
            )[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

    for i in range(len(instances)):
        instances_dict = {"image_id": img_id, "id": i}
        if instances.has("scores"):
            instances_dict["score"] = float(instances.scores[i])
        if instances.has("pred_boxes"):
            instances_dict["bbox"] = boxes[i].tolist()
        if instances.has("pred_classes"):
            instances_dict["category_id"] = instances.pred_classes[i].tolist()
        if instances.has("pred_masks"):
            instances_dict["segmentation"] = rles[i]
        if instances.has("pred_keypoints"):
            instances_dict["keypoints"] = instances.pred_keypoints[i]

        json_data.append(instances_dict)

    with open(output_file, "w") as f:
        json.dump(json_data, f)


class Detectron2:
    """Predict images using detectron2 models.

    Methods:
        predict(np.ndarray) -> Instances

    Attributes:
        dataset_metadata (detectron2.data.Metadata): The dataset metadata.
    """

    def __init__(
        self,
        model_config: str,
        model_weights: str,
        confidence_threshold: float = 0.5,
        use_cuda: bool = False,
    ):
        """Create and load a detectron2 model.

        Args:
            model_config (str): Path to the detectron2 configuration yaml.
            model_weights (str): Path or URL to the model weights file.
            confidence_threshold (float, optional): Minimum
                detection confidence of the instances. Defaults to 0.5.
            use_cuda (bool, optional): Execute the model on a cuda device.
                Defaults to False.
        """
        self._conf_thr = confidence_threshold
        self._setup_cfg(model_config, model_weights, use_cuda)
        self._predictor = DefaultPredictor(self.cfg)

        self.dataset_metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )

    def predict(self, image: np.ndarray) -> Instances:
        """Predict an image.

        Args:
            image (np.ndarray): A BGR uint8 image of shape (H, W, 3).

        Returns:
            Instances: The predicted instances.
        """
        predictions = self._predictor(image)
        return predictions["instances"].to("cpu")

    def _setup_cfg(self, config_path: str, model_weights: str, use_cuda: bool = False):
        """Create the model cfg.

        Args:
            config_path (str): Path to the detectron2 configuration yaml.
            model_weights (str): Path or URL to the model weights file.
            use_cuda (bool, optional): Execute the model on a cuda device.
                Defaults to False.
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(str(config_path))
        self.cfg.MODEL.WEIGHTS = str(model_weights)
        if use_cuda:
            if torch.cuda.is_available():
                self.cfg.MODEL.DEVICE = "cuda"
                logger.info(f"Cuda device: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning(f"Cuda is not available!")
                self.cfg.MODEL.DEVICE = "cpu"
        else:
            self.cfg.MODEL.DEVICE = "cpu"
        logger.info(f"Using device: {self.cfg.MODEL.DEVICE}")
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self._conf_thr
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._conf_thr
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self._conf_thr
        self.cfg.freeze()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        "-w",
        required=True,
        help="Path to a model weights",
        type=Path,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a directory with images",
        type=Path,
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory",
        type=Path,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true"
    )
    return parser


def predict_path(model: Detectron2, input_path: Path, output_path: Path):
    logger.info(f"Output path: {output_path}")
    output_path.mkdir(exist_ok=True, parents=True)

    image_path = list(input_path.iterdir())
    logger.info(f"Image paths: {image_path}")

    for i, image_path in enumerate(image_path):
        output_file = output_path / image_path.name
        assert image_path != output_file
        logger.info(f"{image_path} > {output_file}")

        # Predict
        image = cv2.imread(str(image_path))
        instances = model.predict(image)

        # Export JSON
        export_json(
            instances=instances,
            img_id=i,
            output_file=output_path / (image_path.stem + ".json"),
        )

        # Visualize instances
        visualizer = Visualizer(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            metadata=model.dataset_metadata,
            instance_mode=ColorMode.IMAGE,
        )
        visualizer.draw_instance_predictions(instances)
        visualizer.get_output().save(str(output_file))


if __name__ == "__main__":
    args = get_parser().parse_args()

    det2 = Detectron2(
        Path(args.config),
        Path(args.weights),
        confidence_threshold=args.confidence_threshold,
        use_cuda=args.use_cuda,
    )

    predict_path(det2, args.input, args.output)
