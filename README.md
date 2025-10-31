🧠 3D Object Reconstruction from 2D Images

This project reconstructs a 3D point cloud from a single 2D image using deep learning–based monocular depth estimation (MiDaS) and Open3D visualization.
It works entirely in Google Colab or on your local system with minimal setup — no API keys or paid tools required.

📸 Overview

Traditional 3D reconstruction requires multiple images or stereo vision.
This project uses a single RGB image and estimates scene depth using the MiDaS model, which has been trained on diverse datasets to infer relative depth.
The resulting depth map is then converted into a 3D point cloud, visualized interactively using Open3D.

🧩 Features

✅ Single-image 3D reconstruction (no multi-view setup)
✅ Uses MiDaS DPT-Large model (PyTorch Hub)
✅ Depth estimation + visualization pipeline
✅ Open3D-based interactive 3D viewer
✅ Fully runnable in Google Colab / Jupyter Notebook

🛠️ Tech Stack
Component	Library / Framework
Language	Python 3
Deep Learning	PyTorch
Depth Estimation	MiDaS (DPT_Large)
Image Processing	OpenCV, NumPy
Visualization	Open3D, Matplotlib
📦 Installation & Setup
🔹 Run on Google Colab

Simply upload the notebook (Untitled107.ipynb) to Google Colab and run all cells — dependencies install automatically.

🔹 Run locally

Clone the repository:

git clone https://github.com/<your-username>/3d-object-reconstruction.git
cd 3d-object-reconstruction


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)


Install required packages:

pip install torch torchvision torchaudio opencv-python open3d matplotlib timm pillow numpy

🧮 How It Works

Load Pre-trained MiDaS Model

Fetches DPT_Large from intel-isl/MiDaS via PyTorch Hub.

Input a 2D Image

You can upload your own image or use a sample one.

Estimate Depth

MiDaS predicts relative depth for every pixel.

Generate 3D Point Cloud

Depth map is projected into 3D space using camera intrinsics.

Visualize in 3D

Render interactively using Open3D.

🧠 Example Results
Input 2D Image	Estimated Depth Map	Reconstructed 3D Point Cloud

	
	

(Replace with your actual screenshots from output cells.)

🧾 File Structure
📁 3d-object-reconstruction/
├── Untitled107.ipynb      # Main Jupyter notebook
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
└── sample_images/         # (Optional) Your test images

🧪 Example Code Snippet
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

📚 References

MiDaS: Monocular Depth Estimation (Intel ISL)

Open3D Library

PyTorch

OpenCV

👨‍💻 Author

Name: Pranesh, Erlapally Manohar Kritik, Gopi Anand 
Project Title: 3D Object Reconstruction from 2D Images
Company: Valise
Year: 2025 Submission
