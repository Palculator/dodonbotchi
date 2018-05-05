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
        'click==6.7',
        'cycler==0.10.0',
        'Jinja2==2.10',
        'kiwisolver==1.0.1',
        'MarkupSafe==1.0',
        'matplotlib==2.2.2',
        'numpy==1.14.2',
        'pandas==0.22.0',
        'Pillow==5.1.0',
        'pyparsing==2.2.0',
        'python-dateutil==2.7.2',
        'pytz==2018.4',
        'scikit-learn==0.19.1',
        'six==1.11.0',
        'tensorflow==1.8.0',
        'keras-rl==0.4.0'
    ]
)
