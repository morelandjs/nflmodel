from setuptools import setup

setup(
    name='nflmodel',
    version='0.1',
    description='NFL point spread and point total predictions.',
    author='J. Scott Moreland',
    author_email='morelandjs@gmail.com',
    packages=['nflmodel'],
    scripts=['scripts/nflmodel'],
    package_data={'nflmodel': ['data/betting_lines.csv']},
    install_requires=[
        'hyperopt',
        'matplotlib',
        'melo @ git+https://git@github.com/morelandjs/melo.git@dev#egg=melo',
        'nflgame_redux @ git+https://git@github.com/morelandjs/nflgame.git@master#egg=nflgame_redux',
        'numpy',
        'pandas',
        'scipy >= 0.18.0',
        'setuptools',
        'SQLAlchemy',
    ]
)
