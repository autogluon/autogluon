import os
import subprocess
import sys
from setuptools import setup
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Custom command to invoke full_install.sh or full_install.bat during installation."""

    def run(self):
        # Run the standard install process
        install.run(self)

        # Determine the appropriate installation script based on the OS
        if os.name == "nt":  # Windows
            script = "full_install.bat"
        else:  # Unix-based (Linux, macOS)
            script = "full_install.sh"

        script_path = os.path.join(os.path.dirname(__file__), script)

        if os.path.exists(script_path):
            print(f"Running {script}...")
            try:
                # Run the script
                result = subprocess.run(
                    [script_path],
                    shell=True,
                    check=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"{script} failed with exit code {result.returncode}")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running {script}: {e}")
                sys.exit(1)
        else:
            print(f"{script} not found. Skipping custom installation step.")


setup(
    name="autogluon",
    version="0.1.0",
    description="Forked version of AutoGluon",
    author="Saeid",
    author_email="saeid@example.com",
    license="Apache-2.0",
    url="https://github.com/Saeidjamali/autogluon",
    cmdclass={
        "install": CustomInstallCommand,
    },
)
