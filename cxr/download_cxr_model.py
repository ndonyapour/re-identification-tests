import os
from huggingface_hub import snapshot_download, login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))
elif os.getenv("HUGGINGFACE_HUB_TOKEN"):
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
else:
    # Interactive login (will prompt for token)
    print("Please provide your Hugging Face token.")
    print("You can get it from: https://huggingface.co/settings/tokens")
    print("Or set it as environment variable: export HF_TOKEN=your_token_here")
    token = input("Enter your Hugging Face token: ").strip()
    if token:
        login(token=token)
    else:
        raise ValueError("Authentication required. Please provide a Hugging Face token.")

# Download the model (requires access approval first)
print("Downloading CXR Foundation model...")
print("Note: You must have access approval for google/cxr-foundation repository")
print("Request access at: https://huggingface.co/google/cxr-foundation")

snapshot_download(
    repo_id="google/cxr-foundation",
    local_dir='./cxr_models',
    allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*']
)

print("Download complete!")
