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
        'melo',
        'numpy',
        'matplotlib',
        'scipy >= 0.18.0',
        'joblib',
        'nflgame_redux >= 2.0.1a1',
        'pandas',
        'setuptools',
        'SQLAlchemy',
        'hyperopt'
    ]
)
