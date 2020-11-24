import pathlib

from setuptools import find_packages, setup

base = pathlib.Path(__file__).parent.resolve()
long_description = (base / "README.md").read_text(encoding="utf-8")


setup(name="imbDRL",
      version="2020.11.24.1",
      author="Thijs van den Berg",
      author_email="denbergvanthijs@gmail.com",
      description="Imbalanced Classification with Deep Reinforcement Learning.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Denbergvanthijs/imbDRL",
      packages=find_packages(),
      classifiers=["Programming Language :: Python :: 3.8",
                   "License :: OSI Approved :: Apache Software License",
                   "Operating System :: OS Independent",
                   "Environment :: GPU :: NVIDIA CUDA :: 10.1",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence"],
      keywords="imbalanced classification, deep reinforcement learning, deep q-network, reward-function, classification, medical",
      python_requires=">=3.8")
