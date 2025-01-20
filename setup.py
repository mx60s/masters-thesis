# thesis/setup.py

from setuptools import setup, find_packages

def main():
    setup(
        name="predictive_coding",
        version="0.1.0",
        packages=find_packages(),
        install_requires=[
            "numpy",
            "torch",
            "torchvision",
            "xformers",
            "Pillow",
            "matplotlib",
            "seaborn",
            "scipy",
            "tqdm",
            "scikit-learn",
        ],
    )

if __name__ == "__main__":
    main()
