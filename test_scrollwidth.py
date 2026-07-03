import asyncio
from playwright.async_api import async_playwright
from tests.unit.test_product_carousel_browser import _many_products, _carousel_harness_html
from app.templating import render_product_carousel

async def main():
    carousel_html = render_product_carousel(_many_products())
    html = _carousel_harness_html(carousel_html)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 375, "height": 812})
        await page.set_content(html)
        await asyncio.sleep(1)
        layout = await page.evaluate(
            """() => {
              const doc = document.documentElement;
              const body = document.body;
              return {
                docScrollWidth: doc.scrollWidth,
                docClientWidth: doc.clientWidth,
                bodyScrollWidth: body.scrollWidth,
                bodyClientWidth: body.clientWidth,
                innerWidth: window.innerWidth,
              };
            }"""
        )
        print(layout)
        await browser.close()

asyncio.run(main())
