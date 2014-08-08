from setuptools import setup, Extension
from Cython.Build import cythonize

def readme():
    with open('README.rst') as f:
        return f.read()

#ext_modules = [Extension('sini_inference_fns',['sini_inference_fns.pyx'])]

setup(name = "obliquity",
      version = "0.1",
      description = "Infer the stellar obliquity distribution of transiting planet systems.",
      long_description = readme(),
      author = "Timothy D. Morton",
      author_email = "tim.morton@gmail.com",
      url = "https://github.com/timothydmorton/obliquity",
      packages = ['obliquity'],
      ext_modules = cythonize(["obliquity/*.pyx"]),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      install_requires=['cython','pandas','simpledist'],
      zip_safe=False
) 
