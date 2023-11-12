from setuptools import setup, find_packages

setup(
    name='basereduce',
    version='0.1.0',
    description='A library for knowledge extraction from foundation computer vision models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Luxonis',
    #author_email='your.email@example.com',
    url='https://github.com/luxonis/basereduce',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'torch',
        'transformers',
        'diffusers',
        'tqdm',
        'Pillow',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='computer vision AI machine learning generative models',
)
