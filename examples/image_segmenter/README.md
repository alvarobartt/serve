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

Then, once TorchServe is deployed and the model is registered, you can start sending requests to the REST API in order
to use the model's inference for an input image. In this case a Python example on how to send requests to an 
Image Segmentation model registered in TorchServe is being presented.

First we will start with the preprocessing part, as it means the preparation of the input data in order to fit the
model requirements. In this example, we just need to transform the input image into bytes, but a `preprocessing()` 
function is being presented so that anyone can easily add more steps to the preprocessing part, but take into consideration
that for image models, the transformation into bytes is a must.

```python
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def preprocessing(img_path_or_buf):
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

    # Transform the PIL.Image into a bytes string (required for the inference)
    raw_image_bytes = BytesIO()
    raw_image.save(raw_image_bytes, format="PNG")
    raw_image_bytes.seek(0)

    return raw_image_bytes.read()
```

__Note that the preprocessing should be more extense, but in this case as the model has been deployed with a pre-defined 
handler named `image_segmenter`, there is no need to apply any transformation to the input image besides the conversion to bytes
as that's the expected input format of images in the requests data.__

Then you will need to proceed with the request to `localhost:8080`, which are the default address and port where TorchServe is
deployed. So the request should be sent to the `predictions/` endpoint followed by the name of the registered model that will be
used for the inference over the input image. This means that the resulting url is: `localhost:8080/predictions/fcn`.

```python
import requests
import torch

def predict(preprocessed_image_bytes):
    # Send HTTP Post request to TorchServe Inference API
    url ="localhost:8080/predictions/fcn"
    req = requests.post(url, data=preprocessed_image_bytes)

    # Convert the output list into a torch.Tensor
    output = req.json()
    return torch.FloatTensor(output)
```

For more information regarding the TorchServe REST APIs please visit: https://pytorch.org/serve/rest_api.html)