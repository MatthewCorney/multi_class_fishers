from setuptools import setup, find_packages

setup(name='multi_class_fishers',
      version='0.0.1',
      description='Python implementation fishers test for mult class problems',
      url='https://github.com/MatthewCorney',
      author='Matthew Corney',
      author_email='matthew_corney@yahoo.co.uk',
      license='MIT',
      package_dir={'':"src"},
      packages=find_packages("src"),
      install_requires=[
          'numpy', 'scipy','python-constraint',
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False)
