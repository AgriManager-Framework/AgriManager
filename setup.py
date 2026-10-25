from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README_PATH = ROOT / "README.md"

setup(
    name="agrimanager",
    version="0.1.0",
    description="AgriManager: a framework and benchmark for language-model crop management across simulators",
    long_description=README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "",
    long_description_content_type="text/markdown",
    author="AgriManager Contributors",
    python_requires=">=3.12",
    packages=find_packages(include=("agrimanager", "agrimanager.*")),
    include_package_data=True,
    install_requires=[
        "pyyaml>=6.0.3",
        "tqdm>=4.67.1",
        "openai>=2.6.1",
        "google-genai>=1.47.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords="llm agriculture reinforcement-learning simulation",
    license="Apache-2.0",
    project_urls={
        "Homepage": "https://github.com/AgriManager-Framework/AgriManager",
    },
)
