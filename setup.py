"""Setup configuration for SSD-VLM package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ssd-vlm",
    version="0.1.0",
    author="Research Team",
    description="Simple Self-Distillation for Vision Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ssd-vlm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # torch/torchvision/torchaudio는 서버 사전 설치 버전 사용
        "transformers>=4.51.0",
        "peft>=0.11.0",
        "deepspeed>=0.14.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.9.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "accelerate>=0.27.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.13.0",
        ],
    },
)
