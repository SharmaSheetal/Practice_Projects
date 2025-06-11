import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

__version__ = "0.0.0"
REPO_NAME = "NLP"  
AUTHOR_USER_NAME = "SharmaSheetal"
SRC_REPO = "textSummarizer"
Author_Email = "sharmasheetal9798@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=Author_Email,
    description="A small package for text summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # Link to your GitHub repo
    project_urls={  # Additional project links
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},  # Source code is in the src directory
    packages=setuptools.find_packages(where="src"),  # Automatically find packages in src
    classifiers=[
        "Programming Language :: Python :: 3",
       "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version required
)