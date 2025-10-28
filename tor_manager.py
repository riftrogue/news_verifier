"""
Automated Tor Management for News Verifier
Handles Tor installation, startup, and connection without GUI
"""

import os
import sys
import time
import socket
import logging
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import requests

logger = logging.getLogger(__name__)

class TorManager:
    """Manages Tor daemon lifecycle automatically"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / ".tor"
        self.tor_executable = None
        self.tor_process = None
        self.socks_port = 9050  # Standard Tor SOCKS port
        self.control_port = 9051  # Standard Tor control port
        
    def ensure_tor_installed(self) -> bool:
        """Download and install Tor if not present"""
        try:
            # Check if Tor is already available
            if self._find_existing_tor():
                return True
                
            # Create data directory
            self.data_dir.mkdir(exist_ok=True)
            
            # Download Tor expert bundle
            logger.info("📥 Downloading Tor daemon...")
            tor_url = "https://archive.torproject.org/tor-package-archive/torbrowser/14.5.8/tor-expert-bundle-windows-x86_64-14.5.8.tar.gz"
            
            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                response = requests.get(tor_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📥 Downloading: {progress:.1f}%", end="", flush=True)
                
                print()  # New line after progress
                tmp_path = tmp_file.name
            
            # Extract Tor
            logger.info("📦 Extracting Tor...")
            import tarfile
            with tarfile.open(tmp_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            
            # Find tor executable
            tor_dirs = list(self.data_dir.glob("**/Tor"))
            if tor_dirs:
                tor_exe = tor_dirs[0] / "tor.exe"
                if tor_exe.exists():
                    self.tor_executable = str(tor_exe)
                    logger.info(f"✅ Tor installed: {self.tor_executable}")
                    
            # Cleanup
            os.unlink(tmp_path)
            
            return self.tor_executable is not None
            
        except Exception as e:
            logger.error(f"❌ Failed to install Tor: {e}")
            return False
    
    def _find_existing_tor(self) -> bool:
        """Find existing Tor installation"""
        # Check common paths
        common_paths = [
            r"C:\Program Files\Tor Browser\Browser\TorBrowser\Tor\tor.exe",
            str(self.data_dir / "Tor" / "tor.exe"),
            "tor.exe"  # In PATH
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.tor_executable = path
                logger.info(f"🔍 Found existing Tor: {path}")
                return True
        
        # Check if tor is in PATH
        try:
            subprocess.run(["tor", "--version"], capture_output=True, check=True)
            self.tor_executable = "tor"
            logger.info("🔍 Found Tor in PATH")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return False
    
    def start_tor(self) -> bool:
        """Start Tor daemon"""
        try:
            if not self.tor_executable:
                logger.error("❌ Tor executable not found")
                return False
            
            # Check if Tor is already running
            if self._is_port_open(self.socks_port):
                logger.info(f"✅ Tor already running on port {self.socks_port}")
                return True
            
            # Create Tor config
            config_path = self.data_dir / "torrc"
            config_content = f"""
# Tor configuration for news verifier
SocksPort {self.socks_port}
ControlPort {self.control_port}
DataDirectory {self.data_dir / 'tor_data'}
Log notice stdout
"""
            config_path.write_text(config_content.strip())
            
            # Create data directory
            (self.data_dir / 'tor_data').mkdir(exist_ok=True)
            
            # Start Tor process
            logger.info("🚀 Starting Tor daemon...")
            self.tor_process = subprocess.Popen([
                self.tor_executable,
                "-f", str(config_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            
            # Wait for Tor to start
            for i in range(30):  # Wait up to 30 seconds
                if self._is_port_open(self.socks_port):
                    logger.info(f"✅ Tor started successfully on port {self.socks_port}")
                    return True
                time.sleep(1)
                print(f"\r⏳ Waiting for Tor to start... ({i+1}/30)", end="", flush=True)
            
            print()  # New line
            logger.error("❌ Tor failed to start within 30 seconds")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start Tor: {e}")
            return False
    
    def _is_port_open(self, port: int) -> bool:
        """Check if port is open"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                return result == 0
        except:
            return False
    
    def stop_tor(self):
        """Stop Tor daemon"""
        if self.tor_process:
            logger.info("🛑 Stopping Tor daemon...")
            self.tor_process.terminate()
            try:
                self.tor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tor_process.kill()
            self.tor_process = None
    
    def cleanup(self):
        """Alias for stop_tor() for compatibility"""
        self.stop_tor()
    
    def get_proxy_url(self) -> Optional[str]:
        """Get SOCKS proxy URL if Tor is running"""
        if self._is_port_open(self.socks_port):
            return f"socks5://127.0.0.1:{self.socks_port}"
        return None
    
    def __enter__(self):
        """Context manager entry"""
        if self.ensure_tor_installed() and self.start_tor():
            return self
        raise RuntimeError("Failed to initialize Tor")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_tor()


def test_tor_manager():
    """Test function to verify Tor manager works"""
    try:
        with TorManager() as tor:
            proxy_url = tor.get_proxy_url()
            print(f"✅ Tor proxy available at: {proxy_url}")
            
            # Test the proxy
            import requests
            session = requests.Session()
            session.proxies = {'http': proxy_url, 'https': proxy_url}
            
            # Get IP without Tor
            direct_ip = requests.get('https://httpbin.org/ip', timeout=10).json()['origin']
            
            # Get IP with Tor
            tor_ip = session.get('https://httpbin.org/ip', timeout=10).json()['origin']
            
            print(f"🌐 Direct IP: {direct_ip}")
            print(f"🔒 Tor IP: {tor_ip}")
            
            if direct_ip != tor_ip:
                print("✅ Tor is working correctly - IP addresses differ!")
                return True
            else:
                print("⚠️ Warning: IP addresses are the same")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the Tor manager
    test_tor_manager()