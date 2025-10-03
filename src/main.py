import os
import torch

from unzip_images import process_zip_images
from train import train_on_batch
from utils import chunk_list, get_input_with_timeout
import constants

def main() -> None:
    """
    Trains a cnn on the images from the zip files, in batches.
    After each batch (zip file), waits for user input to continue, stop and save or exit without saving.
    This was done due to computing power, in order to stop anytime the user wishes.
    """
    model = None

    for batch_name, image_paths in process_zip_images():
        print(f"Processing {batch_name} with {len(image_paths)} images")

        for chunk_idx, chunk_paths in enumerate(chunk_list(image_paths, 500), 1):
            print(f" Training chunk {chunk_idx} with {len(chunk_paths)} images")

            model, save_flag = train_on_batch(chunk_paths, model=model)

            if not save_flag:
                print("Aborting training without saving.")
                return

        cont = get_input_with_timeout("Continue with next batch? (Y/n/c): ", 20).strip().lower()

        if cont == "n":
            print("Saving model and exiting.")
            os.makedirs(os.path.dirname(constants.MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), constants.MODEL_PATH)
            return
        elif cont == "c":
            print("Exiting without saving.")
            return


if __name__ == "__main__":
    main()

