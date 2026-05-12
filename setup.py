from setuptools import setup, find_packages

setup(
    name="elgin",
    version="1.0.0",
    author="Dr Takshak Shende",
    author_email="takshak.shende@gmail.com",
    description=(
        "ELGIN: Eulerian-Lagrangian Graph Interaction Network -- "
        "a physics-informed GNN surrogate for particle-laden turbulent flow simulation"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TakshakShende/ELGIN",
    project_urls={
        "Source": "https://github.com/TakshakShende/ELGIN",
        "Paper":  "https://github.com/TakshakShende/ELGIN",
    },
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pillow>=9.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
