from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

this_file = Path(__file__).resolve()
readme = this_file.parent / 'README_EN.md'

setup(
    name='autodl-gpu',
    version='0.1.1',
    description='Automatic Deep Learning, towards fully automated multi-label classification for image, video, text, speech, tabular data.',
    package_data={'': ['README_EN.md']},
    long_description=readme.read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    author='DeepWisdom',
    author_email='autodl@fuzhi.ai',
    url='https://github.com/DeepWisdom/AutoDL',
    download_url='https://github.com/DeepWisdom/AutoDL/archive/v1.0.0.tar.gz',
    keywords=["autodl", "automl", "nas", "feature-engineering", "model-selection", "full-automl", "artificial-intelligence", "lightgbm",
              "resnet", "pytorch", "tensorflow", "python", "autodl-challenge", "ai", "deeplearning", "data-science", "machine-learning",
              "big-data", "multi-label", "automated-machine-learning"],
    install_requires=[
        "numpy==1.16.2",
        "pandas==0.24.2",
        "tensorflow-gpu==1.15.0",
        "jupyter==1.0.0",
        "seaborn==0.9.0",
        "scipy==1.2.1",
        "matplotlib==3.0.3",
        "scikit-learn==0.20.3",
        "pyyaml==5.1.1",
        "psutil==5.6.6",
        "h5py==2.9.0",
        "keras==2.2.4",
        "playsound==1.2.2",
        "librosa==0.7.1",
        "protobuf==3.7.1",
        "xgboost==0.90",
        "pyyaml==5.1.1",
        "lightgbm==2.2.3",
        "torch==1.3.1",
        "torchvision==0.4.2",
        "jieba==0.39",
        "nltk==3.4.5",
        "spacy==2.1.6",
        "gensim==3.8.0",
        "polyglot==16.7.4",
        "hyperopt==0.2.3",
        "catboost==0.21",
        "fastai",
        "kapre==0.1.4",
        "keras-radam",
    ],
    extras_require={
        'tests': ['pytest>=4.4.0',
                  'flake8',
                  'isort',
                  'pytest-xdist',
                  'pytest-cov',
                  'coverage',
                  'typeguard>=2,<3',
                  'typedapi>=0.2,<0.3'
                  ],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache 2.0",
    packages=find_packages(exclude=('tests',)),
    project_urls={
        "Bug Reports": "https://github.com/DeepWisdom/AutoDL/issues",
        "Source": "https://github.com/DeepWisdom/AutoDL",
    },
    python_requires='>=3.6'
)