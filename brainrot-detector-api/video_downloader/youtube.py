import yt_dlp
from .base import BaseScraper
from .utils import setup_logger

logger = setup_logger("YouTubeScraper")

class YouTubeScraper(BaseScraper):
    PLATFORM_NAME = "youtube"
    
    def can_handle(self, query_or_url):
        if not query_or_url.startswith('http'):
            return True
        return any(domain in query_or_url.lower() for domain in [
            'youtube.com', 
            'youtu.be',
            'youtube.com/shorts'
        ])

    def fetch_urls(self, query, output_file, seen_urls=None):
        logger.info(f"[YouTube] Fetching URLs for query: {query} (Limit ~{self.search_limit_override})")
        
        limit = self.search_limit_override
        
        if not query.startswith('http'):
            search_query = f"ytsearch{limit}:{query} shorts"
        else:
            search_query = query
        
        logger.info(f"Running query: {search_query}")
        
        found_urls = []
        with yt_dlp.YoutubeDL(self.extract_opts) as ydl:
            try:
                result = ydl.extract_info(search_query, download=False)
                
                if 'entries' in result:
                    entries = result['entries']
                elif 'url' in result:
                    entries = [result]
                else:
                    entries = []

                for entry in entries:
                    if not entry:
                        continue
                    
                    url = entry.get('url') or entry.get('webpage_url')
                    if not url:
                        continue
                    
                    video_id = self._extract_video_id(entry, url)
                    if video_id:
                        final_url = f"https://www.youtube.com/shorts/{video_id}"
                        
                        duration = entry.get('duration')
                        if duration:
                            if duration <= self.max_duration_seconds:
                                found_urls.append(final_url)
                        else:
                            found_urls.append(final_url)
                            
            except Exception as e:
                logger.error(f"Error fetching YouTube URLs: {e}")

        self._save_urls_to_file(found_urls, query, output_file, seen_urls=seen_urls)
        return found_urls

    def _extract_video_id(self, entry, url):
        video_id = entry.get('id')
        if not video_id:
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in url:
                video_id = url.split("youtu.be/")[1].split("?")[0]
            elif "shorts/" in url:
                video_id = url.split("shorts/")[1].split("?")[0]
        return video_id
