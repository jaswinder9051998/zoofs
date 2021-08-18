from setuptools import setup

setup(name='zoofs',
      version='0.1.0',
      url='https://github.com/jaswinder9051998/zoofs',
      author='JaswinderSingh',
      author_email='jaswinder9051998@gmail.com',
      license='Apache License 2.0',
      packages=['zoofs'],
      zip_safe=True,
	description="zoofs is a Python library for performing feature selection using an variety of nature inspired wrapper algorithms..",
      long_description=open("README.md").read(),
	  install_requires=[
		"pandas",
		"numpy",
		"scipy",
            "plotly"
		],

      )
