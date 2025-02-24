from setuptools import setup, find_packages

setup(
    name="ai-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "gitpython>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-agent=ai_agent.cli:cli",
        ],
    },
    author="carlhannes",
    description="An AI coding agent implementing Learn-by-interact principles",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)