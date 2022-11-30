from base64 import b64decode
from PIL import Image
from io import BytesIO

from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.codecs.string import StringRequestCodec, StringCodec
from mlserver.logging import logger

from torchvision.transforms import PILToTensor
from torchvision.models import resnet50, ResNet50_Weights

class CustomModel(MLModel):

  async def load(self) -> bool:
        
    self._weights = ResNet50_Weights.DEFAULT
    
    self._model = resnet50(weights=self._weights).eval() 

    self._preprocessor = self._weights.transforms()
     
    self.ready = True
    return self.ready

  async def predict(self, payload: InferenceRequest) -> InferenceResponse:

    # Decode request
    b64_input = StringRequestCodec.decode_request(payload)

    # Base64 to tensor
    tensor_input = [self.b64_string_to_tensor(b64) for b64 in b64_input]

    # Preprocess images
    preprocessed_input = [self.preprocess(img_tensor) for img_tensor in tensor_input]
    
    # Classify images
    outputs = [self.classify(img_tensor) for img_tensor in preprocessed_input]
    # logger.info(outputs)

    # Construct the response
    response = StringRequestCodec.encode_response(
      model_name=self.name,
      model_version=self.version,
      payload=outputs
    )

    return response

  def b64_string_to_tensor(self, b64_string: str):
    # Convert the images from base64 strings to torch tensors 
    img_pil = Image.open(BytesIO(b64decode(b64_string)))
    img_tensor = PILToTensor()(img_pil)
    return img_tensor

  def preprocess(self, img_tensor):
    # Apply inference preprocessing transforms
     return self._preprocessor(img_tensor).unsqueeze(0)

  def classify(self, img_tensor):
    # Use the model and get the predicted category
    prediction = self._model(img_tensor).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = self._weights.meta["categories"][class_id]
    class_string = f"{category_name}: {100 * score:.1f}%"

    return class_string
