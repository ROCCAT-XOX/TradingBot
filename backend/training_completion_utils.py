import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def log_training_completion(symbols, model_path, model_metrics=None, config=None):
    """
    Logge Informationen über den Abschluss des Trainings und erstelle eine Benachrichtigungsdatei.

    Args:
        symbols (list): Liste der trainierten Symbole
        model_path (str): Pfad, in dem die Modelle gespeichert wurden
        model_metrics (dict, optional): Metriken für jedes trainierte Modell
        config (dict, optional): Verwendete Trainingskonfiguration
    """
    # Sicherstellen, dass logs-Verzeichnis existiert
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(parent_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Training-Abschluss-Informationen loggen
    logger.info("=========================================")
    logger.info("Training vollständig abgeschlossen!")
    logger.info(f"Modelle für {len(symbols)} Symbole trainiert: {', '.join(symbols)}")
    logger.info(f"Modelle wurden gespeichert in: {model_path}")
    logger.info("=========================================")

    # Modellmetriken loggen, falls verfügbar
    if model_metrics:
        logger.info("Modell-Metriken:")
        for symbol, metrics in model_metrics.items():
            if metrics:
                rmse = metrics.get('rmse', 'N/A')
                mse = metrics.get('mse', 'N/A')
                logger.info(f"  {symbol}: RMSE={rmse}, MSE={mse}")

    # Benachrichtigungsdatei erstellen
    notification_file = os.path.join(logs_dir, 'training_complete.json')

    # Konfigurationsbereinigung - sensible Daten entfernen
    safe_config = None
    if config:
        safe_config = {k: v for k, v in config.items()
                       if not any(secret in k.lower() for secret in ['secret', 'password', 'key', 'token'])}

    # Benachrichtigungsdaten zusammenstellen
    notification_data = {
        'status': 'completed',
        'timestamp': datetime.now().isoformat(),
        'models': symbols,
        'model_path': model_path,
        'metrics': model_metrics or {},
        'config': safe_config
    }

    # Benachrichtigungsdatei schreiben
    try:
        with open(notification_file, 'w') as f:
            json.dump(notification_data, f, indent=4)
        logger.info(f"Trainingsabschluss-Benachrichtigung gespeichert: {notification_file}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Trainingsabschluss-Benachrichtigung: {e}")


def check_training_completion():
    """
    Überprüft, ob ein Training abgeschlossen wurde, indem nach der Benachrichtigungsdatei gesucht wird.

    Returns:
        dict or None: Benachrichtigungsdaten, falls das Training abgeschlossen ist, sonst None
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notification_file = os.path.join(parent_dir, 'logs', 'training_complete.json')

    if os.path.exists(notification_file):
        try:
            with open(notification_file, 'r') as f:
                notification = json.load(f)

            # Optional: Datei nach dem Lesen löschen
            # os.remove(notification_file)

            return notification
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Trainings-Abschlussbenachrichtigung: {e}")

    return None


def clear_training_notification():
    """Löscht die Trainingsabschluss-Benachrichtigungsdatei."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notification_file = os.path.join(parent_dir, 'logs', 'training_complete.json')

    if os.path.exists(notification_file):
        try:
            os.remove(notification_file)
            logger.info(f"Trainingsabschluss-Benachrichtigung gelöscht: {notification_file}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Löschen der Trainingsabschluss-Benachrichtigung: {e}")

    return False


# Optional: Desktop-Benachrichtigung senden (erfordert zusätzliche Bibliothek 'plyer')
def send_desktop_notification(title, message):
    """
    Sendet eine Desktop-Benachrichtigung.

    Args:
        title (str): Titel der Benachrichtigung
        message (str): Nachrichtentext

    Returns:
        bool: True bei Erfolg, False bei Fehler
    """
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name='Trading-AI Dashboard',
            timeout=10
        )
        return True
    except ImportError:
        logger.warning("Plyer ist nicht installiert. Desktop-Benachrichtigungen nicht verfügbar.")
        return False
    except Exception as e:
        logger.error(f"Fehler beim Senden der Desktop-Benachrichtigung: {e}")
        return False