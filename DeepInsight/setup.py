from setuptools import setup

submodule = 'DeepInsight'
install_requires = [
    'scikit-learn>=0.22',
    'pandas',
    
    f'autogluon.tabular_to_image=={version}',    
]

setup(
    name='DeepInsight',
    version='0.1.0',
    packages=['pyDeepInsight'],
    url='https://github.com/alok-ai-lab/deepinsight',
    license='GPLv3',
    author='Keith A. Boroevich',
    author_email='kaboroevich@gmail.com',
    description='A methodology to transform a non-image data to an image for'
                ' convolution neural network architecture',
    install_requires=install_requires,
    extras_require={
        'ImageTransformer_fit_plot': ['matplotlib']
    }
)
