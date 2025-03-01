#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import webbrowser
import threading
import logging
import argparse
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'dashboards_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_all_dashboards")

# Festlegen der Ports für die verschiedenen Dashboards
PORTS = {
    "main_dashboard": 8501,  # Standard Streamlit-Port
    "simple_dashboard": 8502,
    "ai_dashboard": 8503
}


def run_dashboard(dashboard_file, port, name):
    """Startet ein Dashboard auf einem bestimmten Port."""
    try:
        logger.info(f"Starte {name} auf Port {port}...")

        # Kommando zum Starten des Dashboards
        cmd = [
            "streamlit", "run", dashboard_file,
            "--server.port", str(port),
            "--server.headless", "true"  # Verhindert, dass Browser automatisch geöffnet wird
        ]

        # Starte den Prozess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Warte kurz, um sicherzustellen, dass der Server startet
        time.sleep(2)

        # Öffne den Browser für dieses Dashboard
        webbrowser.open(f"http://localhost:{port}")

        logger.info(f"{name} erfolgreich gestartet!")

        return process
    except Exception as e:
        logger.error(f"Fehler beim Starten von {name}: {e}")
        return None


def start_training(training_mode="default"):
    """Startet das Training des ML-Modells."""
    try:
        logger.info(f"Starte ML-Modell-Training im Modus: {training_mode}...")

        # Kommando zum Starten des Trainings
        cmd = ["python", "backend/train.py"]

        # Füge Trainingsmodus-Parameter hinzu, falls vorhanden
        if training_mode != "default":
            cmd.extend(["--mode", training_mode])

        # Starte den Prozess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        logger.info("ML-Modell-Training gestartet!")

        return process
    except Exception as e:
        logger.error(f"Fehler beim Starten des ML-Modell-Trainings: {e}")
        return None


def start_trading_bot(mode="paper", test_mode=True):
    """Startet den Trading-Bot im angegebenen Modus."""
    try:
        logger.info(f"Starte Trading-Bot im Modus: {mode} (Test-Modus: {test_mode})...")

        # Kommando zum Starten des Trading-Bots
        cmd = ["python", "backend/trade_bot.py", "--mode", mode]

        if test_mode:
            cmd.append("--test")

        # Starte den Prozess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        logger.info(f"Trading-Bot im {mode}-Modus gestartet!")

        return process
    except Exception as e:
        logger.error(f"Fehler beim Starten des Trading-Bots: {e}")
        return None


def log_output(process, name):
    """Liest die Ausgabe eines Prozesses und schreibt sie ins Log."""
    for line in iter(process.stdout.readline, ""):
        if line:
            logger.info(f"{name}: {line.strip()}")

    for line in iter(process.stderr.readline, ""):
        if line:
            logger.error(f"{name}: {line.strip()}")


def main():
    parser = argparse.ArgumentParser(description='Starte alle Trading-AI-Dashboards und Komponenten')
    parser.add_argument('--train', action='store_true', help='Starte das ML-Modell-Training')
    parser.add_argument('--train-mode', type=str, default='default', help='Training-Modus (default, fast, deep)')
    parser.add_argument('--bot', action='store_true', help='Starte den Trading-Bot')
    parser.add_argument('--bot-mode', type=str, default='paper', choices=['paper', 'live', 'test'],
                        help='Trading-Bot-Modus (paper, live, test)')
    parser.add_argument('--main-dash', action='store_true', help='Starte das Hauptdashboard')
    parser.add_argument('--simple-dash', action='store_true', help='Starte das einfache Dashboard')
    parser.add_argument('--ai-dash', action='store_true', help='Starte das AI-Dashboard')
    parser.add_argument('--all', action='store_true', help='Starte alle Komponenten')

    args = parser.parse_args()

    # Wenn keine spezifischen Argumente angegeben sind, alle starten
    if not (args.train or args.bot or args.main_dash or args.simple_dash or args.ai_dash or args.all):
        args.all = True

    # Stelle sicher, dass der logs-Ordner existiert
    os.makedirs('logs', exist_ok=True)

    processes = {}
    threads = {}

    # Training starten, wenn angefordert
    if args.train or args.all:
        processes["training"] = start_training(args.train_mode)
        if processes["training"]:
            threads["training"] = threading.Thread(
                target=log_output,
                args=(processes["training"], "Training"),
                daemon=True
            )
            threads["training"].start()

    # Trading-Bot starten, wenn angefordert
    if args.bot or args.all:
        processes["trading_bot"] = start_trading_bot(args.bot_mode, test_mode=(args.bot_mode == 'test'))
        if processes["trading_bot"]:
            threads["trading_bot"] = threading.Thread(
                target=log_output,
                args=(processes["trading_bot"], "Trading-Bot"),
                daemon=True
            )
            threads["trading_bot"].start()

    # Warte einen Moment, damit Backend-Prozesse starten können
    time.sleep(3)

    # Dashboards starten
    if args.main_dash or args.all:
        processes["main_dashboard"] = run_dashboard(
            "frontend/dashboard.py",
            PORTS["main_dashboard"],
            "Hauptdashboard"
        )

    if args.simple_dash or args.all:
        processes["simple_dashboard"] = run_dashboard(
            "frontend/simple_dashboard.py",
            PORTS["simple_dashboard"],
            "Einfaches Dashboard"
        )

    if args.ai_dash or args.all:
        processes["ai_dashboard"] = run_dashboard(
            "frontend/ai_dashboard.py",
            PORTS["ai_dashboard"],
            "AI-Dashboard"
        )

    logger.info("Alle angeforderten Komponenten wurden gestartet")
    logger.info("Drücke STRG+C, um alle Prozesse zu beenden")

    try:
        # Halte den Hauptprozess am Leben, bis er unterbrochen wird
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Beende alle Prozesse...")
        for name, process in processes.items():
            if process:
                process.terminate()
                logger.info(f"{name} beendet")

        logger.info("Alle Prozesse beendet. Auf Wiedersehen!")


if __name__ == "__main__":
    main()