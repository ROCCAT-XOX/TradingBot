#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Mode Starter für das Trading AI System

Dieses Skript startet alle Komponenten des Trading AI Systems im Trainingsmodus:
1. Trading Bot im Test-Modus (keine echten Trades)
2. Training Dashboard (zentrale Steuerung)
3. Haupt-Dashboard (Chart-Anzeige)
4. KI-Dashboard (Model-Insights)

Alle Dashboards werden in separaten Browser-Tabs geöffnet und
können unabhängig voneinander genutzt werden.
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse
import logging
from datetime import datetime

# Konfiguriere Logging
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f'training_mode_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("training_mode")

# Ports für die verschiedenen Dashboards
PORTS = {
    "training_dashboard": 8500,
    "main_dashboard": 8501,
    "ai_dashboard": 8502,
    "simple_dashboard": 8503
}

# Pfade zu den verschiedenen Python-Dateien
PATHS = {
    "training_dashboard": os.path.join("frontend", "training_dashboard.py"),
    "main_dashboard": os.path.join("frontend", "dashboard.py"),
    "ai_dashboard": os.path.join("ai_dashboard.py"),  # falls in einem anderen Verzeichnis
    "simple_dashboard": os.path.join("frontend", "simple_dashboard.py"),
    "trade_bot": os.path.join("backend", "trade_bot.py"),
    "train": os.path.join("backend", "train.py")
}


def start_dashboard(dashboard_name, port):
    """Startet ein Streamlit-Dashboard auf einem bestimmten Port."""
    try:
        logger.info(f"Starte {dashboard_name} auf Port {port}...")

        # Prüfe, ob die Datei existiert
        dashboard_path = PATHS.get(dashboard_name)
        if not os.path.exists(dashboard_path):
            logger.warning(f"Dashboard-Datei nicht gefunden: {dashboard_path}")

            # Versuche, die Datei im frontend-Verzeichnis zu finden
            alt_path = os.path.join("frontend", os.path.basename(dashboard_path))
            if os.path.exists(alt_path):
                dashboard_path = alt_path
                logger.info(f"Alternative Datei gefunden: {dashboard_path}")
            else:
                logger.error(f"Keine Dashboard-Datei für {dashboard_name} gefunden!")
                return None

        # Kommando zum Starten des Dashboards
        cmd = [
            "streamlit", "run", dashboard_path,
            "--server.port", str(port),
            "--server.headless", "true"  # Browser nicht automatisch öffnen
        ]

        # Starte den Prozess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        return process
    except Exception as e:
        logger.error(f"Fehler beim Starten von {dashboard_name}: {e}")
        return None


def start_trading_bot(test_mode=True):
    """Startet den Trading Bot im Test-Modus."""
    try:
        logger.info(f"Starte Trading Bot im {'Test' if test_mode else 'Paper'}-Modus...")

        # Kommando zum Starten des Bots
        bot_path = PATHS.get("trade_bot")
        cmd = ["python", bot_path, "--mode", "paper"]

        if test_mode:
            cmd.append("--test")

        # Starte den Prozess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        return process
    except Exception as e:
        logger.error(f"Fehler beim Starten des Trading Bots: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Starte das Trading AI System im Trainingsmodus')
    parser.add_argument('--bot', action='store_true', help='Starte den Trading Bot')
    parser.add_argument('--no-test', action='store_true',
                        help='Deaktiviere den Test-Modus (Achtung: kann zu echten Trades führen!)')
    parser.add_argument('--simple', action='store_true', help='Starte auch das einfache Dashboard')

    args = parser.parse_args()

    processes = {}

    print("┌─────────────────────────────────────────────┐")
    print("│ Trading AI System - Trainingsmodus-Launcher │")
    print("└─────────────────────────────────────────────┘")
    print("Starte alle Komponenten...")

    # Starte zuerst das Trading-Dashboard als Hauptkontrollzentrum
    processes["training_dashboard"] = start_dashboard("training_dashboard", PORTS["training_dashboard"])
    if not processes["training_dashboard"]:
        logger.error("Fehler beim Starten des Training-Dashboards! Breche ab.")
        return

    print("- Training Dashboard wird gestartet...")
    time.sleep(3)  # Gib dem Dashboard Zeit zum Starten

    # Öffne das Training-Dashboard im Browser
    webbrowser.open(f"http://localhost:{PORTS['training_dashboard']}")
    print("✓ Training Dashboard geöffnet")

    # Starte den Trading Bot, wenn gewünscht
    if args.bot:
        test_mode = not args.no_test
        processes["trade_bot"] = start_trading_bot(test_mode=test_mode)
        if processes["trade_bot"]:
            print(f"✓ Trading Bot gestartet (Testmodus: {test_mode})")
        else:
            print("✗ Fehler beim Starten des Trading Bots")

    # Starte das Haupt-Dashboard
    processes["main_dashboard"] = start_dashboard("main_dashboard", PORTS["main_dashboard"])
    if processes["main_dashboard"]:
        print("- Haupt-Dashboard wird gestartet...")
        time.sleep(3)
        webbrowser.open(f"http://localhost:{PORTS['main_dashboard']}")
        print("✓ Haupt-Dashboard geöffnet")
    else:
        print("✗ Fehler beim Starten des Haupt-Dashboards")

    # Starte das KI-Dashboard
    processes["ai_dashboard"] = start_dashboard("ai_dashboard", PORTS["ai_dashboard"])
    if processes["ai_dashboard"]:
        print("- KI-Dashboard wird gestartet...")
        time.sleep(3)
        webbrowser.open(f"http://localhost:{PORTS['ai_dashboard']}")
        print("✓ KI-Dashboard geöffnet")
    else:
        print("✗ Fehler beim Starten des KI-Dashboards")

    # Starte optional das einfache Dashboard
    if args.simple:
        processes["simple_dashboard"] = start_dashboard("simple_dashboard", PORTS["simple_dashboard"])
        if processes["simple_dashboard"]:
            print("- Einfaches Dashboard wird gestartet...")
            time.sleep(3)
            webbrowser.open(f"http://localhost:{PORTS['simple_dashboard']}")
            print("✓ Einfaches Dashboard geöffnet")
        else:
            print("✗ Fehler beim Starten des einfachen Dashboards")

    print("\nAlle Komponenten wurden gestartet!")
    print("\nHinweise:")
    print("1. Verwenden Sie das Training Dashboard (Port 8500) zur Steuerung")
    print("2. Das Haupt-Dashboard (Port 8501) zeigt Charts und Portfolio")
    print("3. Das KI-Dashboard (Port 8502) zeigt ML-Modell-Details")
    print()
    print("Drücken Sie STRG+C, um alle Komponenten zu beenden...")

    try:
        # Halte den Hauptprozess am Leben
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nBeende alle Komponenten...")
        for name, process in processes.items():
            if process:
                process.terminate()
                print(f"- {name} beendet")

        print("Alle Komponenten wurden beendet. Auf Wiedersehen!")


if __name__ == "__main__":
    main()