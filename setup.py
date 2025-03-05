from setuptools import setup, find_packages

setup(
    name='cadv',
    version='1.02',
    packages=['cadv', 'cadv.network'],
    install_requires=[
        'numpy',
        'matplotlib',
        'cartopy',
        'paramiko'
        # cualquier otra dependencia
    ],
    author='Christian Dávila',
    author_email='davila.met.92@gmail.com',
)
