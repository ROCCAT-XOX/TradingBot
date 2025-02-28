import os
import sys
import logging
import subprocess
import time

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


def main():
    """Hauptfunktion zum Starten der Trading-Bot-Anwendung."""
    logger.info("Trading AI Projekt wird gestartet...")

    # Verzeichnisstruktur überprüfen
    check_directory_structure()

    # Konfiguration überprüfen
    if not check_config():
        return

    # Dashboard starten
    start_dashboard()


if __name__ == "__main__":
    main()