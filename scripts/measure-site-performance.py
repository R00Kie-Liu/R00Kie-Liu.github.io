#!/usr/bin/env python3
"""Compare cold browser loads of before/after builds; saves raw repeatable evidence."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
from urllib.parse import urlsplit

from playwright.async_api import async_playwright

PAGES = {
    'home': '/', 'blog': '/blog/',
    'attention': '/2026/07/27/attention-residual-mhc-xhc-survey.html',
    'scaling': '/2026/05/06/scaling-infra-ch03.html',
    'mtp': '/2026/10/23/mtp-pretraining-self-drafting.html',
}
OPTIONAL = ('googletagmanager.com', 'google-analytics.com', 'clustrmaps.com', 'giscus.app')
INIT = '''window.__audit={lcp:[],tasks:[]};
new PerformanceObserver(l=>l.getEntries().forEach(e=>__audit.lcp.push(e.startTime))).observe({type:'largest-contentful-paint',buffered:true});
new PerformanceObserver(l=>l.getEntries().forEach(e=>__audit.tasks.push(e.duration))).observe({type:'longtask',buffered:true});'''


async def measure(browser, base, name, path, args):
    context = await browser.new_context(viewport={'width': args.width, 'height': 1000})
    await context.route('**/*', lambda r: r.abort() if any(host in r.request.url for host in OPTIONAL) else r.continue_())
    page = await context.new_page()
    await page.add_init_script(INIT)
    cdp = await context.new_cdp_session(page)
    await cdp.send('Network.enable')
    await cdp.send('Network.setCacheDisabled', {'cacheDisabled': True})
    if args.kbps:
        await cdp.send('Network.emulateNetworkConditions', {'offline': False, 'latency': args.latency,
            'downloadThroughput': args.kbps * 1024 / 8, 'uploadThroughput': args.kbps * 1024 / 8})
    if args.cpu > 1:
        await cdp.send('Emulation.setCPUThrottlingRate', {'rate': args.cpu})
    requests = {}
    errors = []

    def started(event):
        requests[event['requestId']] = {'url': event['request']['url'], 'type': event.get('type'), 'bytes': 0}

    def response(event):
        if event['requestId'] in requests:
            requests[event['requestId']]['status'] = event['response']['status']

    def received(event):
        if event['requestId'] in requests:
            requests[event['requestId']]['bytes'] += event.get('encodedDataLength', 0)

    def finished(event):
        if event['requestId'] in requests:
            requests[event['requestId']]['bytes'] = event['encodedDataLength']

    cdp.on('Network.requestWillBeSent', started)
    cdp.on('Network.responseReceived', response)
    cdp.on('Network.dataReceived', received)
    cdp.on('Network.loadingFinished', finished)
    page.on('pageerror', lambda error: errors.append(str(error)))
    await page.goto(base.rstrip('/') + path, wait_until='load', timeout=60000)
    await page.evaluate('async()=>{if(window.MathJax?.startup?.promise)await MathJax.startup.promise;await document.fonts.ready;}')
    await page.wait_for_timeout(1500)
    data = await page.evaluate('''()=>({nav:performance.getEntriesByType('navigation')[0].toJSON(),
        paint:performance.getEntriesByType('paint').find(e=>e.name==='first-contentful-paint')?.startTime,
        lcp:__audit.lcp.at(-1),longTaskMs:__audit.tasks.reduce((a,b)=>a+b,0),
        math:document.querySelectorAll('mjx-container').length,mathErrors:document.querySelectorAll('mjx-merror').length,
        notes:document.querySelectorAll('.post-card').length,overflow:document.documentElement.scrollWidth>innerWidth})''')
    network = list(requests.values())
    data.update(page=name, base=base, bytes=sum(r['bytes'] for r in network),
        fontBytes=sum(r['bytes'] for r in network if r.get('type') == 'Font'),
        imageBytes=sum(r['bytes'] for r in network if r.get('type') == 'Image'),
        externalRequests=sum(urlsplit(r['url']).netloc != urlsplit(base).netloc for r in network if not any(host in r['url'] for host in OPTIONAL)),
        previewRequests=sum('/assets/link-previews/' in r['url'] for r in network), network=network, errors=errors)
    if errors or data['overflow'] or data['mathErrors']:
        raise AssertionError({key: data[key] for key in ('page', 'errors', 'overflow', 'mathErrors')})
    await context.close()
    return data


async def main(args):
    results = []
    async with async_playwright() as pw:
        options = {'headless': True, 'args': ['--no-sandbox', '--disable-dev-shm-usage']}
        if args.chromium:
            options['executable_path'] = args.chromium
        if os.environ.get('HTTPS_PROXY'):
            options['proxy'] = {'server': os.environ['HTTPS_PROXY'], 'bypass': '127.0.0.1,localhost'}
        browser = await pw.chromium.launch(**options)
        for repeat in range(args.repeat):
            for name, path in PAGES.items():
                if name not in args.pages:
                    continue
                # Alternate order to reduce systematic first-run/cache bias.
                pairs = [('before', args.before), ('after', args.after)]
                for label, base in pairs if repeat % 2 == 0 else reversed(pairs):
                    value = await measure(browser, base, name, path, args)
                    value.update(variant=label, repeat=repeat)
                    results.append(value)
                    print(name, label, 'FCP', round(value['paint']), 'ms;', round(value['bytes'] / 1024), 'KiB', flush=True)
        await browser.close()
    summary = {}
    for name in PAGES:
        if name not in args.pages:
            continue
        summary[name] = {}
        for label in ('before', 'after'):
            rows = [row for row in results if row['page'] == name and row['variant'] == label]
            summary[name][label] = {key: round(statistics.median(row[key] for row in rows), 1)
                for key in ('paint', 'lcp', 'bytes', 'fontBytes', 'imageBytes', 'longTaskMs', 'externalRequests')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'method': {'width': args.width, 'repeats': args.repeat,
        'kbps': args.kbps, 'latency_ms': args.latency, 'cpu_slowdown': args.cpu,
        'cache': 'disabled; fresh contexts; gzip local servers; optional analytics/comments blocked equally'},
        'summary': summary, 'runs': results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--pages', nargs='+', choices=list(PAGES), default=list(PAGES))
    parser.add_argument('--kbps', type=int, default=0)
    parser.add_argument('--latency', type=int, default=100)
    parser.add_argument('--cpu', type=float, default=1)
    parser.add_argument('--chromium', default=None)
    parser.add_argument('--output', type=Path, default=Path('.impeccable/performance/comparison.json'))
    asyncio.run(main(parser.parse_args()))
