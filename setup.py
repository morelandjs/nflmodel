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
        'armchair_analysis @ git+https://github.com/morelandjs/armchair_analysis.git@master#egg=armchair_analysis',
        'hyperopt',
        'matplotlib',
        'melo @ git+https://git@github.com/morelandjs/melo.git@dev#egg=melo',
        'numpy',
        'pandas',
        'scipy >= 0.18.0',
        'setuptools',
    ]
)
