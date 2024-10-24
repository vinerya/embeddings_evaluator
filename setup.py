from setuptools import setup, find_packages
from pathlib import Path

# read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="embeddings_evaluator",
    version="1.0.0",
    description="Tool for analyzing and comparing embedding models through pairwise cosine similarity distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Moudather Chelbi",
    author_email="moudather.chelbi@gmail.com",
    url="https://github.com/vinerya/embeddings_evaluator",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "plotly>=5.14.1",
        "scipy>=1.10.1",
        "faiss-cpu",
        "matplotlib"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="embeddings, similarity, evaluation, faiss, visualization",
    include_package_data=True,
    zip_safe=False,
)
