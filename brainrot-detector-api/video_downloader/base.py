import os
import time
import random
from abc import ABC, abstractmethod
import yt_dlp
from .utils import setup_logger

logger = setup_logger("BaseScraper")

# Path to cookies file for authenticated scraping (fallback in current dir)
COOKIES_FILE = os.path.join(os.getcwd(), "cookies.txt")

# Ensure Deno JS runtime is findable by yt-dlp (needed for YouTube signature decoding)
DENO_PATH = os.path.join(os.path.expanduser("~"), ".deno", "bin")
if os.path.exists(DENO_PATH) and DENO_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = DENO_PATH + os.pathsep + os.environ.get("PATH", "")
    logger.info(f"Added Deno to PATH: {DENO_PATH}")


class BaseScraper(ABC):
    """
    Abstract base class for all platform-specific scrapers.
    Provides shared functionality for downloading and filtering videos.
    """
    
    PLATFORM_NAME = "base"
    
    def __init__(self, cookies_from_browser=None, search_limit_override=None, 
                 max_duration_seconds=60, download_archive="downloaded.txt"):
        self.search_limit_override = search_limit_override or 200
        self.max_duration_seconds = max_duration_seconds
        self.download_archive = download_archive
        
        # Custom logger to route yt-dlp messages through Python logging
        # instead of printing directly to stdout (which causes garbled terminal)
        class YDLLogger:
            def debug(self, msg):
                # yt-dlp sends download progress as debug messages
                if msg.startswith('[download]'):
                    logger.info(msg)
            def info(self, msg):
                logger.info(msg)
            def warning(self, msg):
                logger.warning(msg)
            def error(self, msg):
                logger.error(msg)

        self.ydl_opts = {
            # ── FORMAT SELECTION ──────────────────────────────────────────────
            # Download the best available quality WITHOUT re-encoding.
            # The ML pipeline resizes frames to 224×224 internally (EfficientNet),
            # so downloading at native resolution preserves original codec quality
            # and avoids lossy ffmpeg transcoding artifacts that degrade features.
            'format': 'bv*+ba/b / best',
            # Do NOT force merge_output_format — let yt-dlp keep the native
            # container (webm/mp4/mkv) to avoid re-encoding through ffmpeg.
            # If a merge IS needed (separate video+audio streams), use stream
            # copy to mux without transcoding.
            'outtmpl': '%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,                # suppress progress bars entirely
            'logger': YDLLogger(),             # route all output through Python logging
            'match_filter': self._duration_filter,
            'ignoreerrors': True,
            'download_archive': self.download_archive,
            # Suppress ffmpeg verbose output that floods terminal and causes hangs
            'external_downloader_args': {'ffmpeg': ['-loglevel', 'error']},
            # Use stream copy when merging to prevent lossy re-encoding
            'postprocessor_args': {'merger': ['-c', 'copy', '-loglevel', 'error']},
            # Timeout to prevent indefinite hangs on slow/stuck connections
            'socket_timeout': 30,
            # Speed: no sleep between downloads (rely on socket_timeout for hangs)
            'sleep_interval': 0,
            'max_sleep_interval': 1,
            # Retry on errors
            'extractor_retries': 3,             # retry failed extractions
            'retries': 5,                       # retry failed downloads
            'fragment_retries': 5,              # retry failed fragments
            'concurrent_fragment_downloads': 8, # download 8 fragments at once (faster)
        }
        self.extract_opts = {
            'extract_flat': True,
            'quiet': True,
            'ignoreerrors': True,
        }
        
        # Load cookies from browser if specified
        if cookies_from_browser:
            self.ydl_opts['cookiesfrombrowser'] = (cookies_from_browser, None, None, None)
            self.extract_opts['cookiesfrombrowser'] = (cookies_from_browser, None, None, None)
            logger.info(f"Using cookies from browser: {cookies_from_browser}")
        
        # Load cookies from file if available (fallback or additive)
        elif os.path.exists(COOKIES_FILE):
            self.ydl_opts['cookiefile'] = COOKIES_FILE
            self.extract_opts['cookiefile'] = COOKIES_FILE
            logger.info(f"Loaded cookies from {COOKIES_FILE}")

    def _duration_filter(self, info_dict, *, incomplete):
        """Filter out videos that exceed the maximum duration."""
        duration = info_dict.get('duration')
        if duration and duration > self.max_duration_seconds:
            return 'Video is too long'
        return None

    @abstractmethod
    def fetch_urls(self, query, output_file, seen_urls=None):
        """
        Search for videos and save their URLs to a text file.
        Must be implemented by each platform-specific scraper.
        
        Args:
            query (str): Search query, hashtag, or URL.
            output_file (str): Path to save the discovered URLs.
            seen_urls (set, optional): Set of already-found URLs for deduplication.
        """
        pass

    @abstractmethod
    def can_handle(self, query_or_url):
        """
        Check if this scraper can handle the given query or URL.
        
        Args:
            query_or_url (str): The query or URL to check.
            
        Returns:
            bool: True if this scraper can handle it.
        """
        pass

    def _load_archive_ids(self):
        """Load already-downloaded video IDs from the download archive file."""
        archive_ids = set()
        archive_file = self.ydl_opts.get('download_archive')
        if archive_file and os.path.exists(archive_file):
            with open(archive_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            archive_ids.add(parts[1])
        return archive_ids

    def _extract_id_from_url(self, url):
        """Extract video ID from a URL."""
        # Handle youtube.com/watch?v=ID, youtube.com/shorts/ID, youtu.be/ID
        import re
        patterns = [
            r'(?:youtube\.com/(?:watch\?v=|shorts/|embed/|v/))([a-zA-Z0-9_-]{11})',
            r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        # Fallback: last path segment
        return url.rstrip('/').split('/')[-1].split('?')[0]

    def download_videos(self, urls, output_dir="downloads"):
        """
        Download videos from a list of URLs using yt-dlp.
        Pre-filters already-archived URLs and downloads new ones individually.
        
        Args:
            urls (list): List of video URLs.
            output_dir (str): Directory to save downloaded videos. Defaults to "downloads".
        """
        os.makedirs(output_dir, exist_ok=True)
        self.ydl_opts['outtmpl'] = os.path.join(output_dir, '%(id)s.%(ext)s')

        # Pre-filter: skip URLs already in the download archive
        archive_ids = self._load_archive_ids()
        new_urls = []
        skipped = 0
        for url in urls:
            vid_id = self._extract_id_from_url(url)
            if vid_id in archive_ids:
                skipped += 1
            else:
                new_urls.append(url)

        logger.info(f"[{self.PLATFORM_NAME}] {skipped} URLs already in archive (skipped)")
        logger.info(f"[{self.PLATFORM_NAME}] {len(new_urls)} new URLs to download.")
        
        if not new_urls:
            logger.info(f"[{self.PLATFORM_NAME}] Nothing new to download!")
            return

        # Download in batches for speed, but keep batches small so
        # one stuck video doesn't block too many others
        BATCH_SIZE = 10
        total_batches = (len(new_urls) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_num in range(total_batches):
            start = batch_num * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(new_urls))
            batch = new_urls[start:end]
            
            logger.info(f"[{self.PLATFORM_NAME}] Batch {batch_num + 1}/{total_batches} — downloading {len(batch)} videos [{start + 1}-{end}/{len(new_urls)}]")
            
            try:
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    ydl.download(batch)
            except Exception as e:
                logger.error(f"[{self.PLATFORM_NAME}] Batch {batch_num + 1} error: {str(e)}")

            # Brief pause between batches to avoid rate-limiting
            if batch_num < total_batches - 1:
                time.sleep(random.uniform(1, 2))
        
        logger.info(f"[{self.PLATFORM_NAME}] Download complete.")

    def download_from_file(self, file_path, output_dir="downloads"):
        """Helper to read URLs from a file and download."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
            
        with open(file_path, 'r') as f:
            # Filter URLs that this scraper can handle
            all_urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            urls = [url for url in all_urls if self.can_handle(url)]
        
        if urls:
            logger.info(f"[{self.PLATFORM_NAME}] Found {len(urls)} URLs to download")
            self.download_videos(urls, output_dir=output_dir)
        else:
            logger.warning(f"[{self.PLATFORM_NAME}] No compatible URLs found in {file_path}")

    def _save_urls_to_file(self, urls, query, output_file, seen_urls=None):
        """Helper to save discovered URLs to a file, skipping duplicates."""
        if urls:
            # Filter out URLs already seen in previous searches
            if seen_urls is not None:
                new_urls = [u for u in urls if u not in seen_urls]
                seen_urls.update(new_urls)
            else:
                new_urls = urls

            if new_urls:
                with open(output_file, 'a') as f:
                    f.write(f"\n# {self.PLATFORM_NAME.upper()} - {query}:\n")
                    for url in new_urls:
                        f.write(f"{url}\n")
                logger.info(f"[{self.PLATFORM_NAME}] Appended {len(new_urls)} new URLs to {output_file} (skipped {len(urls) - len(new_urls)} duplicates)")
                print(f"[{self.PLATFORM_NAME}] {len(new_urls)} new URLs saved ({len(urls) - len(new_urls)} duplicates skipped)")
            else:
                logger.info(f"[{self.PLATFORM_NAME}] All {len(urls)} URLs for '{query}' were duplicates, skipped.")
        else:
            logger.warning(f"[{self.PLATFORM_NAME}] No URLs found for query: {query}")
