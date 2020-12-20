# Image Segmentation using torchvision's pretrained fcn_resnet_101_coco model.

## Deployment

* Download the pre-trained fcn_resnet_101_coco image segmentation model's state_dict from the following URL :

https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth

```bash
wget https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth
```

* Create a model archive file and serve the fcn model in TorchServe using below commands

    ```bash
    torch-model-archiver --model-name fcn_resnet_101 --version 1.0 --model-file examples/image_segmenter/fcn/model.py --serialized-file fcn_resnet101_coco-7ecb50ca.pth --handler image_segmenter --extra-files examples/image_segmenter/fcn/fcn.py,examples/image_segmenter/fcn/intermediate_layer_getter.py
    mkdir model_store
    mv fcn_resnet_101.mar model_store/
    torchserve --start --model-store model_store --models fcn=fcn_resnet_101.mar
    curl http://127.0.0.1:8080/predictions/fcn -T examples/image_segmenter/fcn/persons.jpg
    ```
* Output
An array of shape [ Batch, Height, Width, 2] where the final dimensions are [class, probability]

```json
[[[[0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993864297866821], [0.0, 0.999385416507721], [0.0, 0.9993811845779419], [0.0, 0.9993740320205688] ... ]]]
```

## Python example

### Preprocessing

First of all, regarding the preprocessing it depends if you deployed the model using `image_segmenter` or `image_classifier` handlers or, on the other hand, you did not used any handler at all. Here you have a piece of code for preprocessing any input image either from path or from buffer, for both approaches, with/without TorchServe handlers.

```python
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def preprocess(img_path_or_buf):
    # Check whether image is a path or a buffer
    raw_image = (
        Image.fromarray(cv2.imread(img_path_or_buf))
        if isinstance(img_path_or_buf, str)
        else img_path_or_buf
    )

    # If buffer was np.array instead of PIL.Image, transform it
    if type(raw_image) == np.ndarray:
        raw_image = Image.fromarray(raw_image)

    # Converts the image to RGB
    raw_image = raw_image.convert("RGB")

    # Here you should uncomment the T.Compose if you are not using 
    # TorchServe `image_classifier` or `image_segmenter` handlers.
    # from torchvision import transforms as T
    # image_processing = T.Compose([
    #     T.Resize(256),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )])
    # raw_image = image_processing(raw_image)

    # Transform the PIL.Image into a bytes string (required for the inference)
    raw_image_bytes = BytesIO()
    raw_image.save(raw_image_bytes, format="PNG")
    raw_image_bytes.seek(0)

    return raw_image_bytes.read()
```

### Inference

Then you will need to proceed with the request to the deployed TorchServe Inference API which by default it's deployed at `localhost:8080` (more information regarding the TorchServe REST APIs available at: https://pytorch.org/serve/rest_api.html)

```python
import requests
import torch

def predict(preprocessed_image_bytes):
    # Send HTTP Post request to TorchServe Inference API
    url ="localhost:8080/predictions/fcn"
    req = requests.post(url, data=preprocessed_image_bytes)
    if req.status_code == 200:
        # Convert the output list into a torch.Tensor
        output = req.json()
        return torch.FloatTensor(output)
    return None
```

### Postprocessing

Then it's up to you how to handle the output of the model, but you can check the default postprocessing behaviors of the `image_segmenter` handler at https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_segmenter.py and the `image_classifier` ones at https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py.