from pathlib import Path
from urllib.request import urlretrieve
import tarfile
import shutil


MODEL_URL = "http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
LABELS_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/object_detection_classes_coco.txt"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_ARCHIVE = MODELS_DIR / "mask_rcnn_inception_v2_coco_2018_01_28.tar.gz"
GRAPH_FILE = MODELS_DIR / "frozen_inference_graph.pb"
CONFIG_FILE = MODELS_DIR / "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
LABELS_FILE = MODELS_DIR / "object_detection_classes_coco.txt"


def download_file(url, destination, message):
    if destination.exists():
        print(f"{message} sudah tersedia.")
        return

    print(f"{message}...")
    urlretrieve(url, destination)


def extract_frozen_graph():
    if GRAPH_FILE.exists():
        print("Pretrained graph sudah tersedia.")
        return

    print("Mengekstrak pretrained graph...")

    graph_member = "mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"

    with tarfile.open(MODEL_ARCHIVE, "r:gz") as tar:
        source = tar.extractfile(graph_member)

        if source is None:
            raise FileNotFoundError("frozen_inference_graph.pb tidak ditemukan dalam archive model.")

        with open(GRAPH_FILE, "wb") as target:
            shutil.copyfileobj(source, target)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    download_file(MODEL_URL, MODEL_ARCHIVE, "Mengunduh pretrained Mask R-CNN")
    extract_frozen_graph()
    download_file(CONFIG_URL, CONFIG_FILE, "Mengunduh konfigurasi Mask R-CNN")
    download_file(LABELS_URL, LABELS_FILE, "Mengunduh label COCO")

    print("Model Mask R-CNN siap digunakan.")


main()