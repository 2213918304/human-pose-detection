"""
人体姿态识别系统安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="human-pose-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="基于 YOLOv8 + MediaPipe 的实时人体姿态检测和分类系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/human-pose-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pose-detect=src.main:main",
        ],
    },
)
