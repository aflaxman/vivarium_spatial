from pathlib import Path

import vivarium_spatial
from vivarium_spatial.constants import metadata

BASE_DIR = Path(vivarium_spatial.__file__).resolve().parent

ARTIFACT_ROOT = Path(
    f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/"
)
