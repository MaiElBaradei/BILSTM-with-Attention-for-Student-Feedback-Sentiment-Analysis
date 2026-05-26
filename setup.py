from setuptools import setup, find_packages

setup(
    name="bilstm-attention",
    version="0.1.0",
    description="BiLSTM + Bahdanau Attention for student feedback sentiment analysis",
    python_requires=">=3.11",
    packages=find_packages(include=["bilstm_attention", "bilstm_attention.*"]),
    install_requires=[
        "torch>=2.2",
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3",
            "pytest-cov>=6.0",
            "mypy>=1.13",
            "black>=24.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "bilstm-train=train:main",
        ],
    },
)
