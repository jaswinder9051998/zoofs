from pathlib import Path

from setuptools import find_packages, setup

# description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


#setup initialization
setup(name='zoofs',
      version='0.1.20',
      url='https://github.com/jaswinder9051998/zoofs',
      author='JaswinderSingh',
      author_email='jaswinder9051998@gmail.com',
      license='Apache License 2.0',
      packages=['zoofs'],
	  description="zoofs is a Python library for performing feature selection using an variety of nature inspired wrapper algorithms..",
      long_description=long_description  ,
	  long_description_content_type='text/markdown',
	  install_requires=["pandas>=1.3.0",
                                "numpy",
                                "scipy>=1.4.1",
                                "plotly>=5.6.0",
                                "colorlog>=6.6.0"],
      include_package_data=True,
      zip_safe=True
      )
