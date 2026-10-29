"""Smoke-test the optimized reading experience with optional external services blocked."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import async_playwright

ATTENTION = '/2026/07/27/attention-residual-mhc-xhc-survey.html'
SCALING = '/2026/05/06/scaling-infra-ch03.html'
MTP = '/2026/10/23/mtp-pretraining-self-drafting.html'


async def settle(page):
    await page.evaluate('async()=>{if(window.MathJax?.startup?.promise)await MathJax.startup.promise;await document.fonts.ready;}')
    await page.wait_for_timeout(80)


async def assert_image_ratio(image):
    await image.evaluate('e=>e.decode().catch(()=>{})')
    size = await image.evaluate('''e=>({naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight,
        width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height,src:e.currentSrc})''')
    assert size['naturalWidth'] > 0 and size['naturalHeight'] > 0, size
    error = abs((size['width'] / size['height']) / (size['naturalWidth'] / size['naturalHeight']) - 1)
    assert error < 0.015, size


async def main(args):
    report = {'viewports': [], 'errors': [], 'bad_responses': []}
    args.output.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        options = {'headless': True, 'args': ['--no-sandbox', '--disable-dev-shm-usage']}
        if args.chromium:
            options['executable_path'] = args.chromium
        if os.environ.get('HTTPS_PROXY'):
            options['proxy'] = {'server': os.environ['HTTPS_PROXY'], 'bypass': '127.0.0.1,localhost'}
        browser = await pw.chromium.launch(**options)
        for width in (1440, 390):
            context = await browser.new_context(viewport={'width': width, 'height': 1000 if width > 600 else 844},
                permissions=['clipboard-read', 'clipboard-write'], has_touch=width < 600)
            await context.route('**/*', lambda r: r.continue_() if r.request.url.startswith(args.base) else r.abort())
            page = await context.new_page()
            page.on('pageerror', lambda e: report['errors'].append(str(e)))
            page.on('response', lambda r: report['bad_responses'].append([r.status, r.url]) if r.status >= 400 else None)
            await page.goto(args.base + '/', wait_until='load')
            await settle(page)
            assert await page.locator('h1').first.inner_text() == 'Huabin Liu'
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            thumbnails = page.locator('.publications img')
            assert await thumbnails.count() >= 20
            assert await thumbnails.evaluate_all("es=>es.every(e=>e.loading==='lazy'&&e.srcset&&e.hasAttribute('width'))")
            assert await thumbnails.evaluate_all("es=>es.every(e=>getComputedStyle(e).objectFit==='contain')")
            await thumbnails.first.scroll_into_view_if_needed()
            await page.wait_for_function('document.querySelector(".publications img").naturalWidth>0')
            await page.screenshot(path=str(args.output / f'home-{width}.png'))
            if width < 600:
                await page.locator('#navMenuBtn').click()
                assert await page.locator('#navDrawer').get_attribute('aria-hidden') == 'false'
                await page.keyboard.press('Escape')
                assert await page.locator('#navDrawer').get_attribute('aria-hidden') == 'true'

            await page.goto(args.base + '/blog/', wait_until='load')
            await settle(page)
            total = await page.locator('.post-card').count()
            assert total >= 44
            await page.get_by_role('button', name='MoE', exact=True).click()
            assert 0 < await page.locator('.post-card:not([hidden])').count() < total
            await page.locator('#blogSearch').fill('no_such_note_983428')
            assert await page.locator('#blogNoResults').is_visible()
            await page.locator('#blogNoResults [data-reset-filters]').click()
            assert await page.locator('.post-card:not([hidden])').count() == total
            await page.locator('.post-card').last.scroll_into_view_if_needed()
            assert await page.locator('.post-card').last.is_visible()
            await page.locator('#blogSearch').scroll_into_view_if_needed()
            assert await page.locator('.post-card-title').first.is_visible()
            await page.wait_for_timeout(100)
            await page.screenshot(path=str(args.output / f'blog-{width}.png'))

            # Verify the real Latin and CJK fonts, including an uncached future
            # character served by the complete self-hosted fallback.
            await page.evaluate('''()=>{const e=document.createElement('p');e.id='font-proof';e.style.fontFamily='var(--font-serif)';e.textContent='Source Serif 思源宋体';document.body.append(e);}''')
            await settle(page)
            cdp = await context.new_cdp_session(page)
            await cdp.send('DOM.enable')
            await cdp.send('CSS.enable')
            document = await cdp.send('DOM.getDocument')
            node = await cdp.send('DOM.querySelector', {'nodeId': document['root']['nodeId'], 'selector': '#font-proof'})
            fonts = (await cdp.send('CSS.getPlatformFontsForNode', {'nodeId': node['nodeId']}))['fonts']
            assert any(f['isCustomFont'] and 'Source Serif' in f['familyName'] for f in fonts), fonts
            assert any(f['isCustomFont'] and 'Noto Serif SC' in f['familyName'] for f in fonts), fonts
            await page.locator('#font-proof').evaluate("e=>e.textContent='龘'")
            await settle(page)
            fallback = (await cdp.send('CSS.getPlatformFontsForNode', {'nodeId': node['nodeId']}))['fonts']
            assert any(f['isCustomFont'] and 'Noto Serif SC' in f['familyName'] for f in fallback), fallback
            await page.locator('#font-proof').evaluate('e=>e.remove()')

            for path, expected_math in ((ATTENTION, 371), (SCALING, 80), (MTP, 33)):
                requests = []
                listener = lambda r: requests.append(r.url)
                page.on('request', listener)
                await page.goto(args.base + path, wait_until='load')
                await settle(page)
                assert await page.locator('mjx-container').count() == expected_math, path
                assert await page.locator('mjx-merror').count() == 0, path
                assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth'), path
                if path == SCALING:
                    assert not any('pointwise-product.gif' in url for url in requests)
                    assert await page.locator('.post-content img[data-deferred-src$="pointwise-product.gif"]').count() == 1
                image = page.locator('.post-image-frame img').first
                await image.scroll_into_view_if_needed()
                await page.wait_for_function('(selector)=>{const e=document.querySelector(selector);return e.complete&&e.naturalWidth>0;}', arg='.post-image-frame img')
                await assert_image_ratio(image)
                await image.click()
                assert await page.locator('.post-image-lightbox').is_visible()
                await page.locator('.post-image-lightbox-img').click()
                assert await page.locator('.post-image-lightbox').is_hidden()
                if path == ATTENTION:
                    for url in ('https://arxiv.org/abs/2512.24880v1', 'https://github.com/MoonshotAI/Kimi-K3'):
                        before = page.url
                        await page.locator('.post-content a[href="' + url + '"]').first.click()
                        await page.wait_for_function('document.querySelector(".link-preview-description").textContent.length>0')
                        assert page.url == before
                        assert await page.locator('.link-preview-open').get_attribute('href') == url
                        assert await page.locator('.link-preview-open').get_attribute('target') == '_blank'
                        await page.locator('.link-preview-copy').click()
                        assert await page.evaluate('navigator.clipboard.readText()') == url
                        await page.keyboard.press('Escape')
                    code = page.locator('.post-code').first
                    expected = await code.locator('pre code').text_content()
                    await code.locator('.code-copy-btn').click()
                    assert await page.evaluate('navigator.clipboard.readText()') == expected
                    assert await page.locator('pre code.hljs').count() > 0
                    assert await page.locator('.back-link').get_attribute('href') == '/blog/'
                if path == SCALING:
                    gif = page.locator('.post-content img[data-full-src$="pointwise-product.gif"]')
                    await gif.scroll_into_view_if_needed()
                    await page.wait_for_function('(selector)=>document.querySelector(selector).src.endsWith("pointwise-product.gif")', arg='img[data-full-src$="pointwise-product.gif"]')
                    await assert_image_ratio(gif)
                    await gif.click()
                    assert (await page.locator('.post-image-lightbox-img').get_attribute('src')).endswith('pointwise-product.gif')
                    await page.keyboard.press('Escape')
                await page.get_by_role('button', name='Toggle theme', exact=True).click()
                await page.wait_for_timeout(350)
                await page.screenshot(path=str(args.output / f'{Path(path).stem}-{width}.png'))
                page.remove_listener('request', listener)
            report['viewports'].append({'width': width, 'notes': total, 'fonts': fonts, 'future_character_fonts': fallback})
            await context.close()

        context = await browser.new_context(java_script_enabled=False, viewport={'width': 390, 'height': 844})
        await context.route('**/*', lambda r: r.continue_() if r.request.url.startswith(args.base) else r.abort())
        page = await context.new_page()
        await page.goto(args.base + SCALING, wait_until='load')
        assert await page.locator('img[data-deferred-src]').is_hidden()
        fallback = page.locator('noscript img[src$="pointwise-product.gif"]')
        await fallback.scroll_into_view_if_needed()
        assert await fallback.is_visible()
        await assert_image_ratio(fallback)
        await context.close()
        await browser.close()
    assert not report['errors'], report['errors']
    assert not report['bad_responses'], report['bad_responses']
    report['checks'] = ['responsive layout', 'mobile menu', 'search and filters', 'full list after scrolling',
        'real fonts and future glyph fallback', 'all representative formulas', 'native and deferred images',
        'image aspect ratios', 'uncropped thumbnails', 'image zoom and restore', 'paper and GitHub previews', 'copy full code and URL', 'themes', 'no-JavaScript fallback']
    (args.output / 'qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('PASS: desktop/mobile reading, fonts, formulas, images, links and preview interactions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:4174')
    parser.add_argument('--chromium', default=None)
    parser.add_argument('--output', type=Path, default=Path('.impeccable/performance/verification'))
    asyncio.run(main(parser.parse_args()))
