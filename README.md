# Invoice rotation

API endpoint for predicting rotation angles of specific invoice documents.


## How to run (Training)

1. Alter `./config.py` file to your liking.
2. Place `.zip` files with images read for training inside `config.ZIP_PATH`.
3. Run:

```{bash}
pip3 install virtualenv
virtualenv .env
source .env/bin/activate
pip3 install -r requirements.txt
python3 src/main.py # This will start training
```

And you'll see the progress of training printed onto `stdout`


## How to run (Getting results via Flask API)

### Option 1. with Python

1. Run:

```{bash}
pip3 install virtualenv
virtualenv .env
source .env/bin/activate
pip3 install -r requirements.txt
python3 api.py # This will run a server on port=5000
```

And in another terminal session:

```{bash}
curl -X POST http://127.0.0.1:5000/predict \
  -F image=@/path/to/image/image.png
```

### Option 2. with Docker

1. Build the dockerimage.
2. Run the container exposing port 5000:

```{bash}
docker build -t invoice-rotation-api .
docker run -p 5000:5000 invoice-rotation-api
```

In another terminal window, run:

```{bash}
curl -X POST http://127.0.0.1:5000/predict \
  -F image=@/path/to/image/image.png
```


## The data & Batching

The data provided as training data for this exercise were the 
following `.zip` files with said contents:

```{stdout}
Processing images-1 with 5000 images                                                                │
Processing images-2 with 4899 images                                                                │
Processing images-3 with 500 images                                                                 │
Processing images-4 with 500 images                                                                 │
Processing images-5 with 1500 images                                                                │
Processing images-6 with 1500 images                                                                │
Processing images-7 with 1500 images                                                                │
Processing images-8 with 1500 images                                                                │
Processing images-9 with 1500 images                                                                │
Processing images-10 with 1500 images                                                               │
```

Due to computing power being an issue and the CNN being trained locally,
the batching strategy was:

1. Batches of `config.BATCH_SIZE` images loaded at a time.
2. For each `.zip` file, every epoch ran on chunks of 500 images printing
benchmarking measures such as average loss (MSE).
3. At the end of each file, the user is prompted if they want to 
continue training or not, and if they want to save the model, or not.


## Transformers and the Model

The transformer chose for this particular exercise was just resizing 
the image and, naturally, transforming it into a Tensor so that the 
CNN can train on it [https://docs.pytorch.org/vision/0.11/transforms.html#](Documentation for torch.nn.transforms).

```{python}
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])
```

The model chosen as a starting point for this exercise was a mobileNetV2
since this is a god compromise between training time and accuracy.

## To do

-   [ ] Randomize which zip file to start training on (`shuffle process_zip_images() output`).
-   [ ] Batch 500 images from each `.zip` file for the model to train on every file.
-   [ ] Abstract `transformers` away to `config.py` for sepration of concerns and consistency.
-   [ ] Abstract `chunk_list(image_paths, 500)` away to `config.py` for sepration of concerns and consistency.


## Out of scope and improvements

-   [ ] A/B test the model against one where `transforms.Resize((image_size, image_size))` is not square.
-   [ ] Rotate images and `sum(transformation_angle, df['angle'])` for more training data.

