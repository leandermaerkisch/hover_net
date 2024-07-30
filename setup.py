from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hover-net",
    version="0.0.7",
    author="Leander Maerkisch",
    author_email="l.maerkisch@gmail.com",
    description="HoVer-Net implementation fork",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leandermaerkisch/hover_net",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "future==0.18.2",
        "imgaug==0.4.0",
        "matplotlib>=3.5.3,<4.0.0",
        "numpy>=1.26.4,<2.0.0",
        "opencv-python>=4.10.0.84,<5.0.0",
        "pandas>=2.2.2,<3.0.0",
        "pillow>=9.0.0,<10.0.0",
        "psutil>=5.7.3,<6.0.0",
        "scikit-image>=0.19.0,<0.20.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "torch>=2.0.0,<3.0.0",
        "torchvision>=0.15.0,<1.0.0",
        "tqdm>=2.2.3,<3.0.0",
        "termcolor>=2.4.0,<3.0.0",
        "openslide-python>=1.3.1,<2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "hover-net-infer=hover_net.run_infer:main",
            "hover-net-train=hover_net.run_train:main",
        ],
    },
)