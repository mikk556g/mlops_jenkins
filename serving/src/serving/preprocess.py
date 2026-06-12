import numpy as np
from PIL import Image
import io


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Takes raw image bytes and returns a normalized numpy array
    ready for ONNX Runtime inference.

    Input:  raw image bytes (JPEG, PNG, etc.)
    Output: float32 array of shape (1, 3, 224, 224)

    Preprocessing matches the val/test transforms used during training:
      Resize to 256x256 -> CenterCrop to 224x224 -> ImageNet normalization
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")                          # ResNet50 expects 3-channel input
    image = image.resize((256, 256), Image.BILINEAR)      # resize to 256x256

    # CenterCrop 224x224
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    array = np.array(image, dtype=np.float32) / 255.0    # scale to [0, 1]

    # ImageNet normalization (must match training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std                          # shape: (224, 224, 3)

    # HWC -> NCHW
    array = array.transpose(2, 0, 1)[np.newaxis, :, :, :]  # shape: (1, 3, 224, 224)

    return array
