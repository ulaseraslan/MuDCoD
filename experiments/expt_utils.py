import sys
import logging
import numpy as np

from pathlib import Path

logging.captureWarnings(True)
MAIN_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = MAIN_DIR / "data"
RESULT_DIR = MAIN_DIR / "results"
sys.path.append(str(MAIN_DIR))

from mudcod.muspces import MuSPCES  # noqa: E402
from mudcod.pisces import PisCES  # noqa: E402
from mudcod.static import Static  # noqa: E402
from mudcod.utils import sutils  # noqa: E402


def log(*args):
    sutils.log(*args)


def get_community_detection_methods(verbose):
    return MuSPCES(verbose=verbose), PisCES(verbose=verbose), Static(verbose=verbose)


def get_msdyn_nw(data_path):
    MSDYN_NW_PATH = data_path / "msdyn_nw.npy"
    msdyn_nw = np.load(MSDYN_NW_PATH)
    return msdyn_nw


def get_msdyn_nw_details(data_path):
    MSDYN_NW_DETAILS_PATH = data_path / "msdyn_nw_details.yaml"
    msdyn_nw_details = sutils.load_yaml(MSDYN_NW_DETAILS_PATH)
    return msdyn_nw_details


def get_adjacency_details(data_path):
    ADJ_DETAILS_PATH = data_path / "adj_details.yaml"
    adj_details = sutils.load_yaml(ADJ_DETAILS_PATH)
    return adj_details


def get_result_path(cell_type, percentile):
    RESULT_PATH = RESULT_DIR / f"{cell_type}/" / f"p{str(percentile)}" / ""
    sutils.safe_create_dir(RESULT_PATH)
    return RESULT_PATH


def ensure_file_dir(path):
    sutils.ensure_file_dir(path)


def get_data_path(cell_type, percentile):
    DATA_PATH = (
        DATA_DIR
        / "cell-type-specific"
        / f"{cell_type}-networks"
        / "adj"
        / f"p{str(percentile)}"
    )
    return DATA_PATH
