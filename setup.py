# -*- coding: utf-8 -*-

from setuptools import setup
import ogres

requirements = [] # Requires tensorflow, but no easy install script exist
version = ogres.__version__


setup(
   name="ogres",
   version=version,
   url="https://github.com/butternutdog/ogres",
   download_url="https://github.com/butternutdog/ogres/tarball/{0}".format(version),
   license="MIT",
   author="Martin Singh-Blom, Jonny Reichwald",
   author_email="ogres@butternutdog.com",
   maintainer="ogres@butternutdog.com",
   description="Thin tensorflow wrapper. Requires tensorflow",
   long_description=None,
   packages=["ogres"],
   install_requires=requirements,
   scripts=[],
   platforms="any",
   zip_safe=True,
   classifiers=[
      "Operating System :: OS Independent",
      "Programming Language :: Python",
      "License :: OSI Approved :: MIT License",
      "Development Status :: 3 - Alpha",
   ]
)
