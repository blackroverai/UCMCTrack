from pathlib import Path
from setuptools import setup, find_packages

# Optional: use README.md as the long description, if present
readme = ""
readme_path = Path(__file__).with_name("README.md")
if readme_path.exists():
    readme = readme_path.read_text(encoding="utf-8")

setup(
    # -------------- project metadata --------------
    name="my_project",           # TODO: change to your package’s name
    version="0.0.1",
    description="Reusable utilities",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.9",

    # -------------- code discovery --------------
    packages=find_packages(exclude=("tests", "docs")),

    # -------------- runtime dependencies --------------
    install_requires=[
        "numpy",
        "filterpy",
        "lap",
        "scipy",
        # argparse is part of the stdlib ≥ 3.2, so only pin it for older Pythons
        "argparse; python_version < '3.2'",
        "ultralytics",
        "opencv-python",
    ],

    # -------------- (optional) metadata classifiers --------------
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
