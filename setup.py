from setuptools import setup, find_packages

setup(
    name="shieldher",
    version="1.0.0",
    description="Personal Safety Monitoring System with AI-powered emotion and voice detection",
    author="ShieldHer Team",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "SpeechRecognition>=3.10.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
        "altair>=5.0.0",
        "pyaudio>=0.2.11",
        "deepface>=0.0.79",
        "tensorflow>=2.13.0"
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'shieldher=shieldher_app:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
