# Video Downloader Module

A modular, standalone library for downloading videos from YouTube, TikTok, and Instagram.

## Features
- **Modular Scraping**: Separated logic for `youtube`, `tiktok`, and `instagram`.
- **Easy Import Interface**: Exposes a simple `get_scraper` factory function.
- **Auto Platform Detection**: `detect_platform` infers the platform based on domain names.
- **Smart Filtering**: Configurable `max_duration_seconds` automatically skips excessive length videos to save bandwidth.
- **Automated Resumption**: Tracks downloaded videos in `downloaded.txt` inside your target folder (configurable) so duplicates are ignored.

## Installation Requirements
You must have the underlying extraction engine `yt-dlp` installed.
```bash
pip install yt-dlp
```

## Basic Usage

```python
from video_downloader import get_scraper, detect_platform

# 1. Define URL
url = "https://www.tiktok.com/@username/video/1234567890"

# 2. Auto-detect platform ('tiktok')
platform = detect_platform(url)

# 3. Instantiate the configured scraper
# Default max_duration_seconds is 60, but you can override it
scraper = get_scraper(platform, max_duration_seconds=60)

# 4. Download it!
scraper.download_videos([url], output_dir="my_downloads")
```
