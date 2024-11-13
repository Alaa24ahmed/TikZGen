from huggingface_hub import HfApi, create_repo
import os
import shutil

def upload_model_with_projector(
    local_model_path,
    repo_id,
    token,
    source_repo="nllg/detikzify-ds-1.3b"
):
    # Initialize Hugging Face API
    api = HfApi(token=token)
    temp_dir = "temp_projector"  # Define temp_dir at the start
    
    # Create the repository if it doesn't exist
    try:
        print(f"Creating/verifying repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            token=token,
            exist_ok=True,
            repo_type="model"
        )
    except Exception as e:
        print(f"Repository creation error: {e}")
        return

    try:
        # First upload your local model files
        print(f"Uploading local model files from {local_model_path}")
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print("Local model files uploaded successfully")

        # Then get and upload the projector folder
        print("Downloading projector folder")
        api.snapshot_download(
            repo_id=source_repo,
            repo_type="model",
            local_dir=temp_dir,
            local_dir_use_symlinks=False
        )
        
        src_projector = os.path.join(temp_dir, "projector")
        if os.path.exists(src_projector):
            print("Uploading projector folder")
            api.upload_folder(
                folder_path=src_projector,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo="projector"
            )
            print(f"Successfully uploaded projector folder to {repo_id}")
        else:
            print("Projector folder not found in source repository")

        print(f"Successfully uploaded everything to {repo_id}")

    except Exception as e:
        print(f"Error during operation: {e}")
        raise  # Re-raise the exception to see the full traceback
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# Example usage
if __name__ == "__main__":
    # Replace these values with your own
    local_model_path = ""
    repo_id = ""
    token = ""  # Get this from https://huggingface.co/settings/tokens
    
    upload_model_with_projector(local_model_path, repo_id, token)