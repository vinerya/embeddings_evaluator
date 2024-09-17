from setuptools import setup, find_packages

setup(
    name='embeddings_evaluator', 
    version='0.4.0',  # Updated version number
    packages=find_packages(include=['embeddings_evaluator', 'embeddings_evaluator.*']),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'plotly',
        'umap-learn'
    ],
    author='Moudather Chelbi',
    author_email='moudather.chelbi@gmail.com',
    description='A package for evaluating and comparing text embeddings using key metrics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vinerya/embeddings_evaluator',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
