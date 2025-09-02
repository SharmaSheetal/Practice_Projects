import setuptools

with open("README.md","r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "hackcu"
AUTHOR_USER_NAME = "divitas"
SRC_REPO = "hackcu"
AUTHOR_EMAIL = "divita.phadakale@colorado.edu"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="package for AI-Assistant",
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)