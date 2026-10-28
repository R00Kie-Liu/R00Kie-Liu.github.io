(function() {
  var content = document.querySelector('.post-content');
  if (!content) return;
  var destinations = new WeakMap();

  function eligible(link) {
    var href = link.getAttribute('href') || '';
    if (!href || href.charAt(0) === '#') return null;
    if (link.hasAttribute('download') || link.hasAttribute('data-no-preview') ||
        link.querySelector('img') || link.closest('.post-image-frame')) return null;
    try {
      var url = new URL(link.href, location.href);
      if (url.protocol !== 'https:' && url.protocol !== 'http:') return null;
      // Internal navigation should follow the original link without an extra step.
      if (url.origin === location.origin) return null;
      return url;
    } catch (error) { return null; }
  }

  content.querySelectorAll('a[href]').forEach(function(link) {
    var url = eligible(link);
    if (!url) return;
    destinations.set(link, url);
    link.classList.add('link-preview-trigger');
    link.setAttribute('aria-haspopup', 'dialog');
    link.setAttribute('aria-controls', 'post-link-preview');
    link.setAttribute('aria-expanded', 'false');
  });

  var card = document.createElement('div');
  card.id = 'post-link-preview';
  card.className = 'link-preview';
  card.hidden = true;
  card.setAttribute('role', 'dialog');
  card.setAttribute('aria-labelledby', 'post-link-preview-title');
  var nativePopover = typeof card.showPopover === 'function';
  if (nativePopover) card.setAttribute('popover', 'manual');

  var header = document.createElement('div');
  header.className = 'link-preview-header';
  var title = document.createElement('p');
  title.id = 'post-link-preview-title';
  title.className = 'link-preview-title';
  var close = document.createElement('button');
  close.type = 'button';
  close.className = 'link-preview-close';
  close.setAttribute('aria-label', '关闭链接预览');
  close.innerHTML = '<i class="fas fa-xmark" aria-hidden="true"></i>';
  header.appendChild(title);
  header.appendChild(close);
  var meta = document.createElement('p');
  meta.className = 'link-preview-meta';
  var description = document.createElement('p');
  description.className = 'link-preview-description';
  var address = document.createElement('p');
  address.className = 'link-preview-address';
  var actions = document.createElement('div');
  actions.className = 'link-preview-actions';
  var open = document.createElement('a');
  open.className = 'link-preview-open';
  open.target = '_blank';
  open.rel = 'noopener noreferrer';
  open.textContent = '打开链接';
  var copy = document.createElement('button');
  copy.type = 'button';
  copy.className = 'link-preview-copy';
  copy.textContent = '复制地址';
  copy.setAttribute('aria-live', 'polite');
  actions.appendChild(open);
  actions.appendChild(copy);
  card.appendChild(header);
  card.appendChild(meta);
  card.appendChild(description);
  card.appendChild(address);
  card.appendChild(actions);
  document.body.appendChild(card);

  var active = null;
  var rectIndex = 0;
  var frame = null;
  var copyTimer = null;

  function kind(url) {
    if (/\.pdf$/i.test(url.pathname) || (url.hostname === 'arxiv.org' && /^\/pdf\//.test(url.pathname))) return 'PDF 文档';
    if (url.hostname === 'arxiv.org' && /^\/(abs|html)\//.test(url.pathname)) return 'arXiv 论文';
    if (url.hostname === 'github.com') return 'GitHub 页面';
    if (url.hostname === 'huggingface.co') return 'Hugging Face 页面';
    return '网页';
  }

  function dismiss(restoreFocus) {
    if (!active) return;
    var previous = active;
    active = null;
    previous.setAttribute('aria-expanded', 'false');
    if (nativePopover && card.matches(':popover-open')) card.hidePopover();
    card.hidden = true;
    clearTimeout(copyTimer);
    if (restoreFocus && previous.isConnected) previous.focus({ preventScroll: true });
  }

  function position() {
    frame = null;
    if (!active) return;
    var rects = active.getClientRects();
    var rect = rects[Math.min(rectIndex, rects.length - 1)];
    if (!rect || rect.bottom <= 56 || rect.top >= innerHeight) {
      dismiss(false);
      return;
    }
    card.style.maxHeight = Math.max(120, Math.min(480, innerHeight - 88)) + 'px';
    if (address.scrollHeight > address.clientHeight + 1) {
      address.tabIndex = 0;
      address.setAttribute('role', 'region');
      address.setAttribute('aria-label', '完整链接地址');
    } else {
      address.removeAttribute('tabindex');
      address.removeAttribute('role');
      address.removeAttribute('aria-label');
    }
    var box = card.getBoundingClientRect();
    var left = Math.max(16, Math.min(rect.left, innerWidth - box.width - 16));
    var top = rect.bottom + 10;
    if (top + box.height > innerHeight - 16) top = rect.top - box.height - 10;
    top = Math.max(72, Math.min(top, innerHeight - box.height - 16));
    card.style.left = left + 'px';
    card.style.top = top + 'px';
  }

  function schedulePosition() {
    if (active && frame === null) frame = requestAnimationFrame(position);
  }

  content.addEventListener('click', function(event) {
    if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    var link = event.target.closest('a[href]');
    var url = link && destinations.get(link);
    if (!url) return;
    event.preventDefault();
    if (event.detail && window.getSelection().toString()) return;
    if (active === link) { dismiss(true); return; }
    dismiss(false);
    active = link;
    rectIndex = 0;
    Array.prototype.forEach.call(link.getClientRects(), function(rect, index) {
      if (event.clientX >= rect.left && event.clientX <= rect.right &&
          event.clientY >= rect.top && event.clientY <= rect.bottom) rectIndex = index;
    });
    title.textContent = link.dataset.previewTitle || link.textContent.trim() || url.hostname;
    meta.textContent = kind(url) + ' · ' + url.hostname;
    description.textContent = link.dataset.previewDescription || link.getAttribute('title') || '';
    description.hidden = !description.textContent;
    address.textContent = url.href;
    open.href = url.href;
    open.setAttribute('aria-label', '在新标签页打开：' + title.textContent);
    copy.textContent = '复制地址';
    copy.removeAttribute('title');
    link.setAttribute('aria-expanded', 'true');
    card.hidden = false;
    if (nativePopover) card.showPopover();
    if (link.getBoundingClientRect().bottom <= 56) link.scrollIntoView({ block: 'center', behavior: 'instant' });
    position();
    open.focus({ preventScroll: true });
  });

  close.addEventListener('click', function() { dismiss(true); });
  open.addEventListener('click', function() { dismiss(false); });
  copy.addEventListener('click', async function() {
    var source = active;
    if (!source) return;
    clearTimeout(copyTimer);
    try {
      await navigator.clipboard.writeText(destinations.get(source).href);
      if (active !== source) return;
      copy.textContent = '已复制';
    } catch (error) {
      if (active !== source) return;
      copy.textContent = '复制失败';
      copy.title = '请选中网址后手动复制';
    }
    copyTimer = setTimeout(function() {
      copy.textContent = '复制地址';
      copy.removeAttribute('title');
    }, 1800);
  });
  document.addEventListener('pointerdown', function(event) {
    if (active && !card.contains(event.target) && !active.contains(event.target)) dismiss(false);
  });
  document.addEventListener('focusin', function(event) {
    if (active && !card.contains(event.target) && event.target !== active) dismiss(false);
  });
  document.addEventListener('keydown', function(event) {
    if (active && event.key === 'Escape') {
      event.preventDefault();
      dismiss(card.contains(document.activeElement) || document.activeElement === active);
    }
  });
  window.addEventListener('scroll', schedulePosition, { passive: true });
  window.addEventListener('resize', schedulePosition);
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(schedulePosition).observe(card);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(schedulePosition);
})();
