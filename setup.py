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
        'hyperopt',
        'matplotlib',
        'melo @ git+https://github.com/morelandjs/elora.git@master#egg=elora'
        'numpy',
        'pandas',
        'scipy >= 0.18.0',
        'setuptools',
    ]
)
