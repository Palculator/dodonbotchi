"""
Gives information about the DoDonBotchi application, specifies its dependencies
and defines entry points for command line use.
"""

from setuptools import setup

setup(
    name='dodonbotchi',
    version='0.1',
    author='Signaltonsalat',
    description='DoDonPatchi AI',
    license='MIT',
    keywords='games bot dodonpatchi ai mame',
    packages=['dodonbotchi'],
    entry_points={
        'console_scripts': [
            'dodonbotchi = dodonbotchi.main:cli'
        ]
    },
    install_requires=[
        'click',
        'cycler',
        'Jinja2',
        'kiwisolver',
        'MarkupSafe',
        'matplotlib',
        'numpy',
        'pandas',
        'Pillow',
        'pyparsing',
        'python-dateutil',
        'pytz',
        'scikit-learn',
        'six',
        'keras-rl',
        'iterfzf',
        'seaborn',
        'shapely',
        'scipy',
    ]
)
