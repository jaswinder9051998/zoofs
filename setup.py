from setuptools import setup
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(name='zoofs',
      version='0.1.4',
      url='https://github.com/jaswinder9051998/zoofs',
      author='JaswinderSingh',
      author_email='jaswinder9051998@gmail.com',
      license='Apache License 2.0',
      packages=['zoofs'],
      zip_safe=True,
	description="zoofs is a Python library for performing feature selection using an variety of nature inspired wrapper algorithms..",
      long_description=long_description  ,
	long_description_content_type='text/markdown',
	  install_requires=[
		"pandas",
		"numpy",
		"scipy",
            "plotly"
		],

      )
