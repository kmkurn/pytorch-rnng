from setuptools import setup, find_packages


setup(name='pytorch-rnng',
      version='0.0.1',
      author='Kemal Kurniawan',
      author_email='kemal@kkurniawan.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      python_requires='>=3.6, <4',
      install_requires=[
          'dill',
          'nltk >=3, <4',
          'torchtext >=0.2, <0.3',
      ])
