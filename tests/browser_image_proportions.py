"""Check article aspect ratios, uncropped thumbnails, and original-image zoom."""
import argparse,asyncio,json,os
from pathlib import Path
from playwright.async_api import async_playwright
PARSER=argparse.ArgumentParser(description='Verify image proportions, uncropped thumbnails, and original-image zoom.')
PARSER.add_argument('--base',default='http://127.0.0.1:4174')
PARSER.add_argument('--chromium',default='/usr/bin/chromium')
PARSER.add_argument('--output',type=Path,default=Path('.impeccable/image-proportions/fixed'))
ARGS=PARSER.parse_args()
BASE=ARGS.base
OUT=ARGS.output;OUT.mkdir(parents=True,exist_ok=True)
PAGES=[('attention','/2026/07/27/attention-residual-mhc-xhc-survey.html',None),('scaling','/2026/05/06/scaling-infra-ch03.html',None),('mtp','/2026/10/23/mtp-pretraining-self-drafting.html',None),('cross-layer','/2026/10/13/cross-layer-kv-reuse.html',None),('small-image','/2026/07/23/moe-visual-generation.html','10-lingbot-video-architecture.png')]
METRIC='''e=>{const s=getComputedStyle(e),r=e.getBoundingClientRect();const w=r.width-parseFloat(s.borderLeftWidth)-parseFloat(s.borderRightWidth)-parseFloat(s.paddingLeft)-parseFloat(s.paddingRight);const h=r.height-parseFloat(s.borderTopWidth)-parseFloat(s.borderBottomWidth)-parseFloat(s.paddingTop)-parseFloat(s.paddingBottom);return {src:e.src,currentSrc:e.currentSrc,fullSrc:e.dataset.fullSrc,alt:e.alt,naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight,attributeWidth:e.getAttribute('width'),attributeHeight:e.getAttribute('height'),renderedWidth:w,renderedHeight:h,cssWidth:s.width,cssHeight:s.height,maxWidth:s.maxWidth,maxHeight:s.maxHeight,objectFit:s.objectFit,aspectRatio:s.aspectRatio,inlineStyle:e.getAttribute('style'),srcset:e.srcset,parentWidth:e.parentElement.getBoundingClientRect().width}}'''
def classify(v):
    nr=v['naturalWidth']/v['naturalHeight'];rr=v['renderedWidth']/v['renderedHeight']
    v['naturalRatio']=nr;v['renderedRatio']=rr;v['ratioErrorPercent']=round(100*(rr/nr-1),2)
    if v['objectFit']=='cover' and abs(rr/nr-1)>.015:
        scale=max(v['renderedWidth']/v['naturalWidth'],v['renderedHeight']/v['naturalHeight'])
        v['visibleSourcePercent']=round(100*v['renderedWidth']*v['renderedHeight']/(scale*scale*v['naturalWidth']*v['naturalHeight']),1)
        v['classification']='cropped; aspect of image content preserved'
    elif v['objectFit']=='contain':v['classification']='contained; aspect of image content preserved'
    elif abs(rr/nr-1)>.015:v['classification']='distorted'
    elif v['renderedWidth']>v['naturalWidth']*1.015:v['classification']='proportionally enlarged'
    else:v['classification']='proportional; original size or reduced'
    return v
async def loaded(image,page):
    await image.scroll_into_view_if_needed()
    await image.evaluate('e=>e.decode().catch(()=>{})')
    await page.wait_for_timeout(80)
async def main():
    report={'pages':[],'errors':[]}
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(executable_path=ARGS.chromium,headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],proxy={'server':os.environ['HTTPS_PROXY'],'bypass':'127.0.0.1,localhost'} if os.environ.get('HTTPS_PROXY') else None)
        context=await browser.new_context()
        await context.route('**/*',lambda r:r.continue_() if r.request.url.startswith(BASE) else r.abort())
        page=await context.new_page();page.on('pageerror',lambda e:report['errors'].append(str(e)))
        for width,height in [(1440,1000),(390,844)]:
            await page.set_viewport_size({'width':width,'height':height})
            await page.goto(BASE+'/',wait_until='load');await page.evaluate('document.fonts.ready')
            photos=page.locator('.image.avatar img,.publications img')
            values=[]
            for i in range(await photos.count()):
                photo=photos.nth(i);await loaded(photo,page)
                v=classify(await photo.evaluate(METRIC));v['role']='avatar' if i==0 else 'publication thumbnail'
                if i==0:assert v['objectFit']=='cover',v
                else:
                    assert v['objectFit']=='contain',v
                    assert await photo.evaluate('e=>{const a=e.closest(".abbr").querySelector("abbr");return !a||a.getBoundingClientRect().top>=e.getBoundingClientRect().bottom-1;}'),v
                values.append(v)
                if i in (1,5):await photo.screenshot(path=str(OUT/f'home-{width}-{i}.png'))
            report['pages'].append({'page':'home','url':BASE+'/','viewport':width,'images':values})
            print('Home',width,[(v['classification'],round(v['renderedWidth'],1),round(v['renderedHeight'],1)) for v in values[:3]],flush=True)
            for name,path,target in PAGES:
                await page.goto(BASE+path,wait_until='load')
                await page.evaluate('async()=>{if(window.MathJax?.startup?.promise)await MathJax.startup.promise;await document.fonts.ready;}')
                images=page.locator('.post-content img') if not target else page.locator('.post-content img[src$="'+target+'"]')
                values=[]
                for i in range(await images.count()):
                    image=images.nth(i);await loaded(image,page)
                    v=await image.evaluate(METRIC)
                    if not v['naturalWidth']:continue
                    v=classify(v)
                    assert v['classification']=='proportional; original size or reduced',v
                    values.append(v)
                    if name=='scaling' and i==0:
                        await image.screenshot(path=str(OUT/f'tpu-in-article-{width}.png'))
                        await image.click()
                        viewer=page.locator('.post-image-lightbox-img');await viewer.evaluate('e=>e.decode()')
                        modal=classify(await viewer.evaluate(METRIC));modal['role']='lightbox';assert modal['objectFit']=='contain',modal;values.append(modal)
                        await page.screenshot(path=str(OUT/f'tpu-lightbox-{width}.png'))
                        await page.keyboard.press('Escape')
                report['pages'].append({'page':name,'url':BASE+path,'viewport':width,'images':values})
                print(name,width,[(v['currentSrc'].rsplit('/',1)[-1],v['naturalWidth'],v['naturalHeight'],round(v['renderedWidth'],1),round(v['renderedHeight'],1),v['classification']) for v in values],flush=True)
        await browser.close()
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    assert not report['errors'],report['errors']
    print('PASS: proportional article images, uncropped thumbnails, labels outside images, and zoom. Report:',OUT/'audit.json',flush=True)
asyncio.run(main())
