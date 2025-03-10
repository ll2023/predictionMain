from setuptools import setup, find_packages

setup(
    name="predictexp",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.2.0',
        'talib-binary>=0.4.0',
        'click>=7.0',
        'pyyaml>=5.4',
        'python-dotenv>=0.19.0',
        'psutil>=5.8.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'plotly>=4.14.0',
    ],
    entry_points={
        'console_scripts': [
            'predictexp=run:main',
        ],
    },
)
