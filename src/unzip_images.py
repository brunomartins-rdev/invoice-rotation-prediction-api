import os
import zipfile
import tempfile
import constants
from typing import Generator, List, Tuple


def process_zip_images() -> Generator[Tuple[str, List[str]], None, None]:
    """
    Looks for zip files in zip folder, and extracts it to a temporary directory.
    Finds all PNG images inside the expected subfolder.
    Yields the base name of the zip file and a list of full image paths.
    """
    zip_files = [f for f in os.listdir(constants.ZIP_PATH) if f.endswith(".zip")]

    for zip_file in zip_files:
        zip_path = os.path.join(constants.ZIP_PATH, zip_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            base_name = os.path.splitext(zip_file)[0]
            expected_folder = os.path.join(temp_dir, base_name)

            if not os.path.isdir(expected_folder):
                continue

            image_paths = []
            for root, _, files in os.walk(expected_folder):
                for file in files:
                    if file.lower().endswith(".png"):
                        image_paths.append(os.path.join(root, file))

            yield base_name, image_paths

