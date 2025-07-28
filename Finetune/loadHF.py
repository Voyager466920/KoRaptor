from huggingface_hub import hf_hub_download
import runpy

# 1) 허브에서 Inference.py 파일만 다운로드
script_path = hf_hub_download(
    repo_id="Voyager466920/KoRaptor_Chatbot",
    filename="Inference.py"
)

# 2) 다운로드한 스크립트 그대로 실행
runpy.run_path(script_path, run_name="__main__")
