import argparse
import os
import sys

# Ensure Python can import the 'video_downloader' package even if executed directly inside the folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from video_downloader import get_scraper, detect_platform
except ImportError as e:
    print(f"Failed to import video_downloader. Ensure you are running this from within a valid directory. Error: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Modularly test a specific video link.")
    parser.add_argument("--url", required=True, help="The video link (e.g., a TikTok URL).")
    parser.add_argument("--platform", default=None, help="Force platform detection (youtube, instagram, tiktok). Auto-detect is default.")
    parser.add_argument("--output", default="test_downloads", help="Directory to deposit the outcome (default: test_downloads).")

    args = parser.parse_args()

    platform = args.platform
    if not platform:
        platform = detect_platform(args.url)
        print(f"[*] Auto-detected platform: '{platform}'")

    try:
        # Override the duration seconds explicitly to ignore the shorts limits when testing general URLs
        scraper = get_scraper(platform, max_duration_seconds=9999)
        print(f"[*] Initiating {scraper.__class__.__name__}...")
        
        print(f"[*] Starting download for: {args.url}")
        scraper.download_videos([args.url], output_dir=args.output)
        print(f"\n[+] Script execution finished! Check the '{args.output}' folder.")
    except Exception as e:
        print(f"[-] Error occurred: {e}")

if __name__ == "__main__":
    main()
