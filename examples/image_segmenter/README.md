# Image Segmentation using torchvision's pretrained fcn_resnet_101_coco model.

## Serve the FCN-ResNet101 model on TorchServe

* Download the pre-trained fcn_resnet_101_coco image segmentation model's state_dict from 
[this URL](https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth) using the 
following command:

    ```bash
    wget https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth
    ```

* Generate the model archive for the FCN-ResNet101 model and move it to the `model_store/` directory using the following command:

    ```bash
    torch-model-archiver --model-name fcn_resnet_101 \
                         --version 1.0 \
                         --model-file examples/image_segmenter/fcn/model.py \
                         --serialized-file fcn_resnet101_coco-7ecb50ca.pth \
                         --handler image_segmenter \
                         --extra-files examples/image_segmenter/fcn/fcn.py,examples/image_segmenter/fcn/intermediate_layer_getter.py

    mkdir model_store
    mv fcn_resnet_101.mar model_store/
    ```

* Start TochServe and register the model archive file using the following commmand:

    ```
    torchserve --start --model-store model_store --models fcn=fcn_resnet_101.mar
    ```

* Run the inference over a sample image using the curl command:

    ```
    curl http://127.0.0.1:8080/predictions/fcn -T examples/image_segmenter/fcn/persons.jpg
    ```

    Which should output an array of shape `[Batch, Height, Width, 2]` where the final dimensions are `[class, probability]`
    and should look like:

    ```json
    [[[[0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993864297866821], [0.0, 0.999385416507721], [0.0, 0.9993811845779419], [0.0, 0.9993740320205688] ... ]]]
    ```

* Install the requirements to run the Python script with the following pip command:

    ```
    pip install requests opencv-python
    ```

* Run the inference over a sample image using this Python script:

    ```python
    from io import BytesIO
    import cv2
    import requests
    import torch

    # Check whether image is a path or a buffer
    raw_image = Image.fromarray(cv2.imread(img_path_or_buf))

    # Converts the image to RGB
    raw_image = raw_image.convert("RGB")

    # Transform the PIL.Image into a bytes string (required for the inference)
    raw_image_bytes = BytesIO()
    raw_image.save(raw_image_bytes, format="PNG")
    raw_image_bytes.seek(0)

    # Read the image as bytes
    input_image = raw_image_bytes.read()

    # Send HTTP Post request to TorchServe Inference API
    url ="http://localhost:8080/predictions/fcn"
    response = requests.post(url, data=input_image)

    # Convert the output list into a torch.Tensor
    output = response.json()
    result = torch.FloatTensor(output)
    ```
