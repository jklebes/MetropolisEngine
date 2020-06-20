import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metropolis-engine",
    version="0.0.1",
    author="Jason Klebes",
    author_email="jsklebes@googlemail.com",
    description="McMC algorithm on mixed real-complex parameter space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jklebes/metropolis-engine",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
