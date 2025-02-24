from setuptools import setup, find_packages

setup(
    name="ai-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=0.27.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "gitpython>=3.1.0",
        "numpy>=1.26.0",
        "pytest>=7.0.0",
        "typing-extensions>=4.0.0",
    ],
)