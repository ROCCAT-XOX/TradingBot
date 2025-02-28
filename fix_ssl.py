"""
Run this script to fix SSL certificate verification issues on macOS.
This will install certificates into your Python environment.
"""

import os
import ssl
import subprocess
import sys


def fix_macos_ssl():
    """Install SSL certificates for macOS Python installations."""
    print("Attempting to fix SSL certificate verification issues...")

    # Check if we're on macOS
    if sys.platform != 'darwin':
        print("This script is only needed on macOS.")
        return

    # Get the Python installation path
    python_path = sys.executable
    python_dir = os.path.dirname(os.path.dirname(python_path))

    # Try to find the Install Certificates.command script
    cert_script = os.path.join(python_dir, 'Resources', 'Install Certificates.command')
    if not os.path.exists(cert_script):
        cert_script = os.path.join(python_dir, 'Install Certificates.command')

    if os.path.exists(cert_script):
        print(f"Found certificate installation script at: {cert_script}")
        try:
            subprocess.run(['bash', cert_script], check=True)
            print("SSL certificates successfully installed!")
        except subprocess.CalledProcessError as e:
            print(f"Error running certificate installation script: {e}")
            manual_fix()
    else:
        print("Certificate installation script not found.")
        manual_fix()


def manual_fix():
    """Apply a manual fix for SSL certificate issues."""
    print("\nAttempting manual fix...")

    try:
        import certifi

        # Set environment variables to use certifi's certificates
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

        print(f"Set SSL_CERT_FILE to: {certifi.where()}")
        print("You should add these lines to your .bash_profile or .zshrc file:")
        print(f"export SSL_CERT_FILE={certifi.where()}")
        print(f"export REQUESTS_CA_BUNDLE={certifi.where()}")

    except ImportError:
        print("certifi package not found. Installing certifi...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'certifi'], check=True)
            print("certifi installed. Please run this script again.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing certifi: {e}")


def test_ssl():
    """Test SSL connection to verify the fix worked."""
    print("\nTesting SSL connection...")

    try:
        import urllib.request
        response = urllib.request.urlopen('https://api.alpaca.markets/v2/account')
        print("SSL connection test successful!")
        return True
    except ssl.SSLCertVerificationError as e:
        print(f"SSL certificate verification still failing: {e}")
        return False
    except Exception as e:
        print(f"Other error during SSL test: {e}")
        return False


if __name__ == "__main__":
    fix_macos_ssl()
    test_ssl()

    print("\nInstructions for your trading application:")
    print("1. Add this to the beginning of your main script (start.py):")
    print("import os")
    print("import certifi")
    print("os.environ['SSL_CERT_FILE'] = certifi.where()")
    print("os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()")
    print("\n2. Or run your application with these environment variables:")
    print("SSL_CERT_FILE=$(python -m certifi) REQUESTS_CA_BUNDLE=$(python -m certifi) python start.py")