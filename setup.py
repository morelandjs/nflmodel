from setuptools import setup

setup(name='nflmodel',
      version='0.1',
      description='NFL point spread and point total predictions.',
      author='J. Scott Moreland',
      author_email='morelandjs@gmail.com',
      packages=['nflmodel'],
      scripts=[
          'scripts/nflmodel-update-data',
          'scripts/nflmodel-train-model',
          'scripts/nflmodel-validate',
      ]
      )
