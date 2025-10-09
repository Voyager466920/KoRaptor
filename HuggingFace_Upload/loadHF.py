from huggingface_hub import hf_hub_download
import runpy

script_path = hf_hub_download(
    repo_id="Voyager466920/KoRaptor_Chatbot",
    filename="Inference.py"
)

runpy.run_path(script_path, run_name="__main__")
