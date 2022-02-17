import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='torchstain',
    version='1.1.0',
    description='Pytorch stain normalization utils',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/EIDOSlab/torchstain',
    author='EIDOSlab',
    author_email='eidoslab@di.unito.it',
    license='MIT',
    packages=find_packages(exclude=('tests')),
    zip_safe=False,
    install_requires=[
        'numpy~=1.19.5',
        'typing-extensions~=3.7.4',
        'torch',
    ],
    extras_require=[
        'tensorflow',
    ]
    python_requires='>=3.6'
)
