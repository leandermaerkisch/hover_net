from setuptools import setup, find_packages

setup(
    name="hover_net",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
    ],
    author="Leander Maerkisch",
    author_email="l.maerkisch@gmail.com",
    description="HoVer-Net implementation fork",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/leandermaerkisch/hover_net",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
