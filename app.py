from flask import Flask, request, jsonify
import os
import subprocess
import gdown
from tqdm import tqdm  # Import tqdm for progress tracking

app = Flask(__name__)

# Directory for the roop model and the face swapper
ROOP_DIR = "/kaggle/working/roop"  # Adjust the path to where your Roop files are
MODEL_PATH = os.path.join(ROOP_DIR, "models/inswapper_128.onnx")

# Helper function to download a file from Google Drive
def download_from_google_drive(url, output_path):
    try:
        file_id = url.split("/d/")[1].split("/view")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, output_path, quiet=False)
        print(f"File downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Failed to download file from {url}. Error: {str(e)}")

# Helper function to check if a package is installed
def is_package_installed(package_name):
    try:
        subprocess.run(["pip", "show", package_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

# Download and setup the models and dependencies with progress bars
def setup_roop():
    os.chdir(ROOP_DIR)

    # Check if the model has already been downloaded
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading inswapper_128.onnx...")
        with tqdm(total=100, desc="Downloading model") as pbar:
            subprocess.run(["wget", "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx", "-O", "inswapper_128.onnx"])
            pbar.update(100)  # Assume the download is 100% when completed
        os.makedirs("models", exist_ok=True)
        subprocess.run(["mv", "inswapper_128.onnx", "./models/"])
    else:
        print("Model is already downloaded.")

    # Check if onnxruntime-gpu is installed
    if is_package_installed("onnxruntime-gpu"):
        print("onnxruntime-gpu is already installed.")
    else:
        print("Installing onnxruntime-gpu...")
        with tqdm(total=100, desc="Installing onnxruntime-gpu") as pbar:
            subprocess.run(["pip", "install", "onnxruntime-gpu"])
            pbar.update(100)

    # Check if torch, torchvision, and torchaudio are installed
    if is_package_installed("torch") and is_package_installed("torchvision") and is_package_installed("torchaudio"):
        print("Torch packages are already installed.")
    else:
        print("Installing torch, torchvision, and torchaudio...")
        with tqdm(total=100, desc="Installing PyTorch") as pbar:
            subprocess.run(["pip", "uninstall", "onnxruntime", "onnxruntime-gpu", "-y"])
            subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio", "--force-reinstall", "--index-url", "https://download.pytorch.org/whl/cu118"])
            pbar.update(100)

# Initialize Roop and dependencies
setup_roop()

@app.route('/swap', methods=['POST'])
def face_swap():
    try:
        os.chdir(ROOP_DIR)

        target_url = request.form.get('target_url')  # Google Drive link for the target video
        source_url = request.form.get('source_url')  # Google Drive link for the source image
        output_path = '/kaggle/working/roop/uploaded_data/outputs/output_face_swap.mp4'  # Default output path

        # Define paths to save the downloaded files
        target_path = "/kaggle/working/roop/uploaded_data/videos/target_video.mp4"
        source_path = "/kaggle/working/roop/uploaded_data/images/source_image.jpg"

        # Ensure directories exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.makedirs(os.path.dirname(source_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Download the files from Google Drive with progress bar
        print("Downloading target video...")
        download_from_google_drive(target_url, target_path)
        print("Downloading source image...")
        download_from_google_drive(source_url, source_path)

        # Run the Roop command to perform face swapping with progress tracking
        print("Performing face swapping...")
        with tqdm(total=100, desc="Face swapping") as pbar:
            command = f"python run.py --target {target_path} --source {source_path} -o {output_path} --execution-provider cuda --frame-processor face_swapper"
            subprocess.run(command, shell=True, check=True)
            pbar.update(100)

        return jsonify({
            'status': 'success',
            'message': 'Face swapping completed',
            'output_path': output_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Test route to check if the API is working
@app.route('/')
def index():
    return "Roop Face Swapping API is running!"

if __name__ == "__main__":
    app.run(port=80)  # Adjust port based on your cloud server configuration
