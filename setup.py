from setuptools import setup, find_packages

setup(
    name='cadv',
    version='0.40',
    packages=['cadv'],
    install_requires=[
        'numpy',
        'matplotlib',
        'cartopy',
        # cualquier otra dependencia
    ],
    author='Christian Dávila',
    author_email='davila.met.92@gmail.com',
)
