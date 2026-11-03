"""Capture the README screenshots from the running app.

Drives the real UI in Chromium rather than mocking anything, so a broken
frontend shows up as a broken screenshot instead of a passing test.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8021"
OUT = Path(__file__).resolve().parent.parent / "screenshots"
OUT.mkdir(exist_ok=True)

EMAIL = "demo@tajurba.pk"
PASSWORD = "tajurba-demo-2026"


def main() -> int:
    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on(
            "console", lambda m: errors.append(m.text) if m.type == "error" else None
        )
        page.on("pageerror", lambda e: errors.append(str(e)))

        # 1. Auth screen
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "01-signin.png"))
        print("01-signin.png")

        # 2. Sign in
        page.fill('#login-form input[name="email"]', EMAIL)
        page.fill('#login-form input[name="password"]', PASSWORD)
        page.click('#login-form button[type="submit"]')
        page.wait_for_selector("#app:not(.hidden)", timeout=15000)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "02-datasets.png"))
        print("02-datasets.png")

        # 3. Open the iris dataset specifically — it is the clearest demo.
        page.click('[data-dataset]:has-text("iris")')
        page.wait_for_selector("#view-dataset:not(.hidden)", timeout=10000)
        page.wait_for_timeout(700)
        page.screenshot(path=str(OUT / "03-dataset.png"))
        print("03-dataset.png")

        # 4. Train
        page.click("#train-btn")
        page.wait_for_selector("#view-run:not(.hidden)", timeout=90000)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "04-leaderboard.png"), full_page=True)
        print("04-leaderboard.png")

        # 5. Predict
        inputs = page.query_selector_all("#predict-form input")
        values = ["5.1", "3.5", "1.4", "0.2"]
        for el, v in zip(inputs, values, strict=False):
            el.fill(v)
        page.click('#predict-form button[type="submit"]')
        page.wait_for_selector("#predict-result:not(.hidden)", timeout=20000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "05-prediction.png"), full_page=True)
        print("05-prediction.png")

        # 6. Mobile
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(BASE, wait_until="networkidle")
        mobile.fill('#login-form input[name="email"]', EMAIL)
        mobile.fill('#login-form input[name="password"]', PASSWORD)
        mobile.click('#login-form button[type="submit"]')
        mobile.wait_for_selector("#app:not(.hidden)", timeout=15000)
        mobile.wait_for_timeout(1200)
        mobile.screenshot(path=str(OUT / "06-mobile.png"), full_page=True)
        print("06-mobile.png")

        # Overflow check at mobile width
        overflow = mobile.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        print(f"mobile horizontal overflow: {overflow}px")

        browser.close()

    if errors:
        print("\nCONSOLE ERRORS:")
        for e in errors:
            print("  ", e)
        return 1

    print("\nno console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
