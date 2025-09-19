"""
Setup script for Qugeister - Quantum Geister AI System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text(encoding='utf-8').strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
else:
    requirements = [
        'torch>=1.11.0',
        'pennylane>=0.28.0',
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'PyYAML>=6.0',
        'tqdm>=4.62.0',
        'pathlib'
    ]

setup(
    name="qugeister",
    version="1.0.0",
    author="Qugeister Development Team",
    author_email="", 
    description="Quantum Geister AI Competition System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qugeister/qugeister",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0"
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "qugeister=qugeister.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "qugeister": [
            "web/templates/*.html",
            "web/static/css/*.css", 
            "web/static/js/*.js",
            "web/static/images/*"
        ]
    },
    project_urls={
        "Bug Reports": "https://github.com/qugeister/qugeister/issues",
        "Source": "https://github.com/qugeister/qugeister",
        "Documentation": "https://qugeister.readthedocs.io/",
    },
)