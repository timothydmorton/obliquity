from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

def readme():
    with open('README.rst') as f:
        return f.read()

sini_inference_fns = [Extension('sini_inference_fns',['obliquity/sini_inference_fns.pyx'],
                                include_dirs=[numpy.get_include()])]

setup(name = "obliquity",
      version = "0.1",
      description = "Infer the stellar obliquity distribution of transiting planet systems.",
      long_description = readme(),
      author = "Timothy D. Morton",
      author_email = "tim.morton@gmail.com",
      url = "https://github.com/timothydmorton/obliquity",
      packages = find_packages(),
      #ext_modules = cythonize(["obliquity/*.pyx"]),
      ext_modules = sini_inference_fns,
      scripts = ['scripts/write_cosi_dist'],
      cmdclass = {'build_ext': build_ext},
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      install_requires=['cython','pandas>=0.13','simpledist'],
      zip_safe=False
) 
