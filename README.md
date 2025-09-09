# KoRaptor 150M
### 150M parameter model *pretrained / fine-tuned from SCRATCH* using *SINGLE GPU*(RTX 3090)
<a href="https://huggingface.co/Voyager466920/KoRaptor_Chatbot" target="_blank">
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=white" alt="Hugging Face Model"/>
</a>
<a href="https://www.youtube.com/watch?v=USPKsNLCRqE&si=AAiD-9Clo-IJnduv" target="_blank">
  <img src="https://img.shields.io/badge/YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white" alt="Youtube video"/>
</a>


## Key Features
- **Language:** Pure Korean  
- **Architecture:** LatentMoE  
- **Parameters:** 150 million
- **Base_model**: Voyager466920/KoRaptor


- **Use case:** Conversational AI / Chatbot  
- **Dataset:** Korean chatbot dataset from AI Hub  
- **License:** Follows the license of the original dataset and model architecture

## Usage
You can easily run inferences using the provided `Inference.py` script. No additional setup is required — simply load the model and start chatting in Korean.  
This model is incompatible with standard transformer loading methods (e.g., AutoModel). For simple inference, use the following code.

```python
from huggingface_hub import hf_hub_download
import runpy

script_path = hf_hub_download(
    repo_id="Voyager466920/KoRaptor_Chatbot",
    filename="Inference.py"
)
```





