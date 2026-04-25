# Instructions For AI Agents

**Target File Path**: `video_downloader/AGENT_INSTRUCTIONS.md`

Agent, if the user asks you to implement video downloading functionality in this project, **do not write bash scripts calling `yt-dlp` directly** and **do not create redundant extraction classes**. 

Instead, leverage this `video_downloader` module.

### How To Use Me Properly
When writing python code to download a video, use the following pattern:

```python
import sys
import os
# Adjust sys.path to ensure 'video_downloader' is accessible if needed

from video_downloader import get_scraper, detect_platform

def grab_video(target_url, destination_dir="downloads"):
    # 1. Identify which scraper to use
    platform = detect_platform(target_url)
    
    # 2. IMPORTANT: If the user says it's a long video, raise the duration! Default is 60s!
    scraper = get_scraper(platform, max_duration_seconds=9999) 
    
    # 3. Use the unified interface arrays
    scraper.download_videos([target_url], output_dir=destination_dir)
```

**Known Quirks:**
- YouTube querying defaults to finding `Shorts`. If standard Youtube videos are requested without limit, adjust durations.
- If Tiktok or Instagram fails silently, it could be a cookie restriction blocking yt-dlp.
