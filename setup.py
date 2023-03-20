import setuptools

__version__ = '0.0.2'

def _parse_requirements(path):
    with open(path) as f:
        return [
            line.rstrip()
            for line in f
            if not (line.isspace() or line.startswith('#'))
        ]

requirements = _parse_requirements('requirements.txt')

setuptools.setup(
    name='tsukuyomichan_talksoft',
    version=__version__,
    packages=['tsukuyomichan_talksoft'],
    package_dir={'tsukuyomichan_talksoft': 'src'},
    install_requires=requirements,
)