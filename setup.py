from setuptools import setup

setup(
    name='nflmodel',
    version='0.1',
    description='NFL point spread and point total predictions.',
    author='J. Scott Moreland',
    author_email='morelandjs@gmail.com',
    packages=['nflmodel'],
    scripts=['scripts/nflmodel'],
    install_requires=[
        'elora @ git+https://github.com/morelandjs/elora.git@master#egg=elora',
        'hyperopt',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy >= 0.18.0',
        'setuptools']
)
