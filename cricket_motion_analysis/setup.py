from setuptools import setup, find_packages

setup(
    name='cricket_motion_analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'mediapipe',
        'opencv-python',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'run_analysis=scripts.run_analysis:main',
        ],
    },
)