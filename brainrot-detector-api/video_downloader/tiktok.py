import yt_dlp
from .base import BaseScraper
from .utils import setup_logger

logger = setup_logger("TikTokScraper")

class TikTokScraper(BaseScraper):
    PLATFORM_NAME = "tiktok"
    
    def can_handle(self, query_or_url):
        return 'tiktok.com' in query_or_url.lower()

    def fetch_urls(self, query, output_file):
        logger.warning("[TikTok] Fetching generic search queries isn't well supported by yt-dlp for TikTok.")
        logger.warning("Please provide an input_file containing a list of direct TikTok URLs instead.")
        return []
