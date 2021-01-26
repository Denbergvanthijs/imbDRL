import pathlib

from setuptools import find_packages, setup

base = pathlib.Path(__file__).parent.resolve()
long_description = (base / "README.md").read_text(encoding="utf-8")
install_requires = (base / "requirements.txt").read_text(encoding="utf-8").split("\n")[:-1]  # Remove empty string at last index

setup(name="imbDRL",
      version="2021.1.26.1",
      author="Thijs van den Berg",
      author_email="denbergvanthijs@gmail.com",
      description="Imbalanced Classification with Deep Reinforcement Learning.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Denbergvanthijs/imbDRL",
      packages=find_packages(),
      classifiers=["Programming Language :: Python :: 3",
                   "License :: OSI Approved :: Apache Software License",
                   "Operating System :: OS Independent",
                   "Environment :: GPU :: NVIDIA CUDA",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence"],
      keywords="imbalanced classification, deep reinforcement learning, deep q-network, reward-function, classification, medical",
      install_requires=install_requires,
      python_requires=">=3.7")
