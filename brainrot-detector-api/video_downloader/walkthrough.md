# Walkthrough: Architecture & Logic

This document maps out how the standalone video scraper is built.

## 1. The Core Engine (`base.py`)
`BaseScraper` serves as the abstract class wrapping `yt_dlp.YoutubeDL`.
- **Options Management**: Pre-loads optimum CLI flags (`'format': 'bv*+ba/b'`) to ensure best audio/video quality and force MP4 output.
- **Filters**: Employs an internal `_duration_filter` allowing developers to bypass abnormally long videos (configured via `max_duration_seconds`).
- **Archive System**: Uses `.download_archive` via `yt-dlp` logic to bypass attempting to download previously downloaded IDs, boosting script rerun performance.

## 2. Platform Implementations (`youtube.py`, `tiktok.py`, `instagram.py`)
- Each platform file inherits `BaseScraper` and overrides `can_handle()`.
- **YouTube**: Enhances normal behavior by formatting shorts correctly and allowing for broad Youtube Query searches.
- **TikTok/Instagram**: Wraps logging handlers specifically around `yt-dlp` direct link integrations since yt-dlp natively resolves their domain targets.

## 3. Exposing Access (`__init__.py`)
- We utilize a dictionary mapping (`SCRAPERS`) to tie strings like `"youtube"` to classes like `YouTubeScraper`.
- The `get_scraper()` factory dynamically provides initialized classes allowing seamless looping across diverse URL lists.
