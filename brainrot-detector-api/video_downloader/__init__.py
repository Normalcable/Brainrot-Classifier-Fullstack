from .base import BaseScraper
from .youtube import YouTubeScraper
from .instagram import InstagramScraper
from .tiktok import TikTokScraper

SCRAPERS = {
    'youtube': YouTubeScraper,
    'instagram': InstagramScraper,
    'tiktok': TikTokScraper,
}

def get_scraper(platform='youtube', **kwargs):
    platform = platform.lower()
    if platform not in SCRAPERS:
        available = ', '.join(SCRAPERS.keys())
        raise ValueError(f"Unknown platform: {platform}. Available: {available}")
    
    return SCRAPERS[platform](**kwargs)

def detect_platform(url_or_query):
    url_or_query_lower = url_or_query.lower()
    if 'instagram.com' in url_or_query_lower or 'instagr.am' in url_or_query_lower:
        return 'instagram'
    if 'tiktok.com' in url_or_query_lower:
        return 'tiktok'
    return 'youtube'

def get_all_scrapers(**kwargs):
    return [scraper_class(**kwargs) for scraper_class in SCRAPERS.values()]

__all__ = [
    'BaseScraper',
    'YouTubeScraper',
    'InstagramScraper',
    'TikTokScraper',
    'get_scraper',
    'detect_platform',
    'get_all_scrapers',
    'SCRAPERS',
]
