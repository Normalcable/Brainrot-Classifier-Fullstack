import yt_dlp
from .base import BaseScraper
from .utils import setup_logger

logger = setup_logger("InstagramScraper")

class InstagramScraper(BaseScraper):
    PLATFORM_NAME = "instagram"
    
    def can_handle(self, query_or_url):
        return any(domain in query_or_url.lower() for domain in [
            'instagram.com', 
            'instagr.am'
        ])

    def fetch_urls(self, query, output_file):
        logger.warning("[Instagram] Fetching generic search queries isn't well supported by yt-dlp for Instagram.")
        logger.warning("Please provide an input_file containing a list of direct Instagram Reel URLs instead.")
        return []
