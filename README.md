# KoRaptor
 
---
language:
- ko
metrics:
- accuracy
- perplexity
base_model:
- Voyager466920/KoRaptor
pipeline_tag: question-answering
---
# KoRaptor 150M
## 150M parameter model *pretrained / fine-tuned from SCRATCH* using *SINGLE GPU*(RTX 3090)


## Key Features
Language: Pure Korean
Architecture: LatentMoE
Parameters: 150 million

Use case: Conversational AI / Chatbot
Dataset: Korean chatbot dataset from AI Hub
License: Follows the license of the original dataset and model architecture

## Usage
You can easily run inference by using the provided inference.py script. No additional setup is required — simply load the model and start chatting in Korean.
This model is not compatible with the standard transformers loading methods (e.g., AutoModel). For simple inference, use the following code.

```python
from huggingface_hub import hf_hub_download
import runpy

script_path = hf_hub_download(
    repo_id="Voyager466920/KoRaptor_Chatbot",
    filename="Inference.py"
)

runpy.run_path(script_path, run_name="__main__")

```
