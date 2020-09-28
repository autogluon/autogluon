from setuptools import setup

install_requires = [
    'numpy',
    'scipy'
    'scikit-learn',
    'pandas',
    'matplotlib'
]

setup(
    name='DeepInsight',
    version='0.1.0',
    packages=['pyDeepInsight'],
    url='https://github.com/alok-ai-lab/deepinsight',
    license='MIT',
    author='Keith A. Boroevich',
    author_email='kaboroevich@gmail.com',
    description='A methodology to transform a non-image data to an image for convolution neural network architecture',
    install_requires=install_requires
)
