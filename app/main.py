import sys
from pathlib import Path

# Add parent directory to path to import from utils
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.anomaly_detection import train_anomaly_model
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('train')

if __name__ == "__main__":
    logger.info("Starting model training...")
    success = train_anomaly_model()
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")
