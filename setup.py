from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="SAFEPG",
    version="1.0.1",
    description="""The frequency-severity model has been widely adopted to analyze highly right-skewed data 
                    in actuarial science. To make the model more interpretable, we expect a predictor has 
                    the same direction of impact on both the frequency and severity. However, the 
                    compotemporary use of the frequence-severity model typically yields inconsistent signs. 
                    To this end, we propose a novel sign-aligned regularization term to facilitate the sign 
                    consistency between the components in the frequency-severity model to enhance interpretability. 
                    We also demonstrate our design of the penalty leads to an algorithm which is quite efficient 
                    in analyzing large-scale data. """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yikai Zhang, Qian Tang, Boxiang Wang",
    packages=find_packages(include=["SAFEPG"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    license="MIT",
    url="https://github.com/YikaiZhang95/SAFE",
)
