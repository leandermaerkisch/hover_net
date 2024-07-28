from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hover-net",
    version="0.0.6",
    author="Leander Maerkisch",
    author_email="l.maerkisch@gmail.com",
    description="HoVer-Net implementation fork",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leandermaerkisch/hover_net",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
        "matplotlib",
        "pandas",
        "openslide-python",
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