import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
import sys
import logging
import subprocess
import time
import argparse

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_directory_structure():
    """Überprüfe, ob die Verzeichnisstruktur korrekt ist und erstelle fehlende Verzeichnisse."""
    directories = [
        'logs',
        'backend/data',
        'backend/models'
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Verzeichnis erstellt: {directory}")


def check_config():
    """Überprüfe, ob die Konfigurationsdatei existiert."""
    config_path = os.path.join('config', 'settings.json')
    if not os.path.exists(config_path):
        logger.error(f"Konfigurationsdatei nicht gefunden: {config_path}")
        logger.info("Bitte stelle sicher, dass du die settings.json Datei im config Verzeichnis hast.")
        return False
    return True


def start_dashboard():
    """Starte das Streamlit Dashboard."""
    try:
        logger.info("Starte das Dashboard...")
        subprocess.run(['streamlit', 'run', 'frontend/dashboard.py'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Fehler beim Starten des Dashboards: {e}")
    except KeyboardInterrupt:
        logger.info("Dashboard manuell beendet.")


def start_websocket():
    """Starte den WebSocket-Listener für Echtzeit-Marktdaten."""
    try:
        logger.info("Starte WebSocket-Integration...")
        subprocess.run(['python', 'backend/websocket_integration.py'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Fehler beim Starten der WebSocket-Integration: {e}")
    except KeyboardInterrupt:
        logger.info("WebSocket-Integration manuell beendet.")


def main():
    """Hauptfunktion zum Starten der Trading-Bot-Anwendung."""
    parser = argparse.ArgumentParser(description='Trading AI Projekt Starter')
    parser.add_argument('--dashboard', action='store_true', help='Starte das Streamlit Dashboard')
    parser.add_argument('--websocket', action='store_true',
                        help='Starte den WebSocket-Listener für Echtzeit-Marktdaten')
    parser.add_argument('--all', action='store_true', help='Starte alle Komponenten')

    args = parser.parse_args()

    logger.info("Trading AI Projekt wird gestartet...")

    # Verzeichnisstruktur überprüfen
    check_directory_structure()

    # Konfiguration überprüfen
    if not check_config():
        return

    if args.all or (not args.dashboard and not args.websocket):
        # Wenn --all angegeben oder keine speziellen Argumente, starte alles
        # Starte WebSocket in einem separaten Prozess
        websocket_process = subprocess.Popen(['python', 'backend/websocket_integration.py'])

        # Gib dem WebSocket Zeit zum Starten
        time.sleep(2)

        # Starte Dashboard (blockierend)
        start_dashboard()

        # Beende WebSocket-Prozess, wenn Dashboard beendet wird
        if websocket_process.poll() is None:  # Wenn der Prozess noch läuft
            websocket_process.terminate()
            websocket_process.wait()
            logger.info("WebSocket-Prozess beendet.")
    else:
        # Starte spezifische Komponenten
        if args.websocket:
            start_websocket()

        if args.dashboard:
            start_dashboard()


if __name__ == "__main__":
    main()