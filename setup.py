from setuptools import setup, find_packages


setup(name='rnng',
      version='0.0.0',
      author='Kemal Kurniawan',
      author_email='kemal@kkurniawan.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'})
