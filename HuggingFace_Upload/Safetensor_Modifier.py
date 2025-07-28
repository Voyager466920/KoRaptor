import torch
from safetensors.torch import save_file

# 1) .pt 로드
state_dict = torch.load(
    r"C:\junha\Git\BFG_2B\HuggingFace_Upload\Finetune\hf_repo\KoRaptor_Base.pt",
    map_location="cpu"
)

# 2) 공유된 뷰 텐서 분리
state_dict["lm_head.weight"]         = state_dict["lm_head.weight"].clone()
state_dict["token_embedding.weight"] = state_dict["token_embedding.weight"].clone()

# 3) safetensors로 저장 (파일명 전체 경로 포함)
save_file(
    state_dict,
    r"C:\junha\Git\BFG_2B\HuggingFace_Upload\Finetune\hf_repo\KoRaptor_Base.safetensors"
)
