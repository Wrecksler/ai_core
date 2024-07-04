from setuptools import setup, find_packages


setup(
    name='ai_core',
    version="0.0.1",
    # url='none',
    author='Wrecksler',
    author_email='wrecksler@gmail.com',
    py_modules=['ai_core'],
    install_requires=[
            'loguru',
            'requests',
            'pyyaml',
            'jinja2',
            'pydantic',
            'cachetools',
            'nltk',
            'pillow',
            'transformers',
    ],
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        '': ["*.yaml", "*.jinja2"],
        },
    include_package_data=True,
)