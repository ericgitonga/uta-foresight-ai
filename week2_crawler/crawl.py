import json
import os
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright

def crawl_headlines(output_path: str = "data/raw_signals.json", target_count: int = 10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    print(f"Launching Playwright Chromium browser to crawl {target_count} public headlines...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        target_url = "https://news.ycombinator.com/"
        print(f"Navigating to {target_url}...")
        page.goto(target_url, wait_until="domcontentloaded", timeout=25000)

        # Extract top stories
        story_elements = page.query_selector_all(".athing")
        for idx, el in enumerate(story_elements[:target_count], start=1):
            title_el = el.query_selector(".titleline > a")
            if not title_el:
                continue

            title = title_el.inner_text().strip()
            url = title_el.get_attribute("href") or ""
            domain = ""
            site_bit = el.query_selector(".sitestr")
            if site_bit:
                domain = site_bit.inner_text().strip()

            results.append({
                "id": idx,
                "title": title,
                "url": url,
                "source": domain or "news.ycombinator.com",
                "crawled_at": datetime.now(timezone.utc).isoformat()
            })

        browser.close()

    print(f"Successfully scraped {len(results)} signals.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved raw signals to {output_path}")
    return results

if __name__ == "__main__":
    signals = crawl_headlines()
    print("\nSample extracted signals:")
    for s in signals[:3]:
        print(f"- [{s['id']}] {s['title']} ({s['source']})")
