from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include
__version__ = '0.0.1dev1'

def _parse_requirements(path):
    with open(path) as f:
        return [
            line.rstrip()
            for line in f
            if not (line.isspace() or line.startswith('#'))
        ]

requirements = _parse_requirements('requirements.txt')
ext = Extension("tsukuyomichan_talksoft", sources=["tsukuyomichan_talksoft.pyx"], include_dirs=['.', get_include()],language_level=3)
setup(
    name='tsukuyomichan_talksoft',
    version=__version__,
    py_modules=['tts_config'],
    install_requires=requirements,
    ext_modules=cythonize([ext])
)