from huggingface_hub import snapshot_download


snapshot_download(
    local_dir_use_symlinks=True,
    repo_type="dataset",
    repo_id="fnlp/character-llm-data",
    local_dir="/work/trainable_agents/data/character-llm-data",
)