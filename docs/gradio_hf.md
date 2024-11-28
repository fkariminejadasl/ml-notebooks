# Hugging Face Guide

## New Model Setup

### Uploading a Model

1. **Create a Model Repository**
   
   Go to Hugging Face and create a `New Model` under your profile, which essentially creates a repository for your models. Refer to [Hugging Face Repositories Getting Started](https://huggingface.co/docs/hub/en/repositories-getting-started) for detailed instructions. 
   
    Once created, you can clone the repository and proceed with the next steps on your local machine.

2. **Set Up Git LFS (Large File Storage)**
   
   Hugging Face uses Git LFS to manage large files, such as model weights. Install and set up Git LFS as follows:
   
   ```bash
   sudo apt install git-lfs
   git lfs install
   ```

3. **Configure Hugging Face Interface**
   
   Install the Hugging Face Hub library and set up authentication:
   
   ```bash
   pip install huggingface_hub
   
   # Configure Git to store credentials
   git config --global credential.helper store
   
   # Log in to Hugging Face CLI
   huggingface-cli login
   
   # Your token will be saved in the following location:
   cat $HOME/.cache/huggingface/token
   
   # Enable LFS support for large files
   huggingface-cli lfs-enable-largefiles
   ```

4. **Push a Large File**
   
   Once Git LFS is configured, you can push large files just like a normal Git commit:
   
   ```bash
   git add your_model
   git commit -m "Add a large file using Git LFS"
   git push
   ```
   
   For more details on uploading models, refer to the [Hugging Face Model Uploading Guide](https://huggingface.co/docs/hub/en/models-uploading).

### Downloading a Model

To download a model from Hugging Face, you can use the following Python code:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="fkariminejadasl/bird", filename="45_best.pth")
```

For more downloading options, refer to the [Hugging Face Models Downloading Guide](https://huggingface.co/docs/hub/en/models-downloading).

---

## Creating a New Space on Hugging Face

To start, navigate to Hugging Face and create a `New Space` under your profile. This will generate a repository for your application. For detailed guidance, refer to the [Gradio Spaces documentation](https://huggingface.co/docs/hub/en/spaces-sdks-gradio).

Once your space is created, you can clone the repository to your local machine, add your `app.py` and `requirements.txt` files, and push the changes back to the repository. The application will launch automatically.

### Application Code and Dependencies

- **`app.py`**: Your Gradio application code must be saved in a file named `app.py`.
  
- **`requirements.txt`**: This file should list all the Python dependencies your app requires, including any custom libraries. An example `requirements.txt` might look like this:

    ```txt
    torch
    gradio
    huggingface_hub
    git+https://github.com/username/repository_name.git@branch_name
    ```

    - Note: The `branch_name` can also be replaced by `tag_name` or `commit_hash` depending on your need.

    If the `pyproject.toml` file for a custom library is located in a subdirectory within the repository, you can specify the subdirectory like this:

    ```txt
    git+https://github.com/username/repository_name.git@branch_name#subdirectory=subdirectory_name
    ```

For further details on how to use `pip install` with various options, refer to the [Pip Install Guide](https://pip.pypa.io/en/stable/cli/pip_install).
