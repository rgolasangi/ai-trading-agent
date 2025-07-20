"""
Main entry point for the AI Trading Agent
"""
import asyncio
import schedule
import time
from datetime import datetime
import signal
import sys

from config.config import Config
from data.data_collector import DataCollectionOrchestrator
from data.data_storage import DataStorageManager
from utils.logger import get_logger

logger = get_logger(__name__)

class TradingAgentMain:
    """Main application class for the AI Trading Agent"""
    
    def __init__(self):
        self.running = False
        self.data_orchestrator = DataCollectionOrchestrator()
        self.storage_manager = DataStorageManager()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def collect_and_store_data(self):
        """Collect and store data"""
        try:
            logger.info("Starting data collection cycle...")
            
            # Collect all data
            collected_data = await self.data_orchestrator.collect_all_data()
            
            if collected_data:
                # Store the data
                success = self.storage_manager.store_collected_data(collected_data)
                if success:
                    logger.info("Data collection and storage cycle completed successfully")
                else:
                    logger.error("Data storage failed")
            else:
                logger.warning("No data collected")
                
        except Exception as e:
            logger.error(f"Error in data collection cycle: {e}")
    
    def schedule_data_collection(self):
        """Schedule periodic data collection"""
        # Schedule data collection every minute during market hours
        schedule.every().minute.do(lambda: asyncio.run(self.collect_and_store_data()))
        
        # Schedule news collection every 5 minutes
        schedule.every(5).minutes.do(lambda: asyncio.run(self.collect_and_store_data()))
        
        logger.info("Data collection scheduled")
    
    async def run_async(self):
        """Main async run loop"""
        try:
            # Validate configuration
            Config.validate_config()
            logger.info("Configuration validated successfully")
            
            # Schedule data collection
            self.schedule_data_collection()
            
            # Initial data collection
            await self.collect_and_store_data()
            
            self.running = True
            logger.info("AI Trading Agent started successfully")
            
            # Main loop
            while self.running:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
            self.running = False
    
    def run(self):
        """Main run method"""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main run: {e}")
        finally:
            logger.info("AI Trading Agent shutdown complete")

def main():
    """Main function"""
    logger.info("Starting AI Trading Agent...")
    
    try:
        app = TradingAgentMain()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start AI Trading Agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

