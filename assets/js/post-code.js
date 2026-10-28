(function() {
  var content = document.querySelector('.post-content');
  if (!content) return;

  var names = { python: 'Python', javascript: 'JavaScript', typescript: 'TypeScript',
    bash: 'Shell', shell: 'Shell', sh: 'Shell', json: 'JSON', yaml: 'YAML',
    html: 'HTML', css: 'CSS', cpp: 'C++', c: 'C', plaintext: 'Text', text: 'Text' };

  content.querySelectorAll('pre').forEach(function(pre, index) {
    var code = pre.querySelector('code');
    if (!code || !code.textContent.trim() || pre.closest('.post-code')) return;
    var source = code.textContent;
    var languageNode = code.closest('[class*="language-"]');
    var match = languageNode && languageNode.className.match(/(?:^|\s)language-([^\s]+)/);
    var language = match ? match[1] : '';
    var languageLabel = names[language] || language || '代码';
    var lineCount = source.replace(/\n$/, '').split('\n').length;

    if (window.hljs) {
      // Highlight the original text once, replacing Jekyll's pre-rendered spans.
      code.textContent = source;
      code.classList.add('language-' + (language && hljs.getLanguage(language) ? language : 'plaintext'));
      hljs.highlightElement(code);
    }

    var wrapper = document.createElement('div');
    wrapper.className = 'post-code';
    var toolbar = document.createElement('div');
    toolbar.className = 'post-code-toolbar';
    var info = document.createElement('span');
    info.textContent = languageLabel + ' · ' + lineCount + ' 行';
    var copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'code-copy-btn';
    copy.textContent = '复制';
    copy.setAttribute('aria-label', '复制完整代码');
    copy.setAttribute('aria-live', 'polite');
    toolbar.appendChild(info);
    toolbar.appendChild(copy);
    var viewport = document.createElement('div');
    viewport.className = 'post-code-scroll';
    var id = 'post-code-' + (index + 1);
    while (document.getElementById(id)) id += '-scroll';
    viewport.id = id;
    var toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'post-code-toggle';
    toggle.hidden = true;
    toggle.setAttribute('aria-controls', id);
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(toolbar);
    wrapper.appendChild(viewport);
    viewport.appendChild(pre);
    wrapper.appendChild(toggle);

    var expanded = false;
    var frame = null;
    var copyTimer = null;
    function update() {
      frame = null;
      var rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      var limit = Math.floor(Math.min(window.innerHeight * 0.55, rem * 28));
      viewport.style.setProperty('--post-code-height', limit + 'px');
      var tall = pre.getBoundingClientRect().height > limit + 1;
      if (!tall) expanded = false;
      wrapper.classList.toggle('is-expanded', expanded);
      toggle.hidden = !tall;
      toggle.textContent = expanded ? '收起代码' : '展开完整代码（' + lineCount + ' 行）';
      toggle.setAttribute('aria-expanded', String(expanded));
      var wide = viewport.scrollWidth > viewport.clientWidth + 1;
      var scrollable = wide || (tall && !expanded);
      if (scrollable) {
        viewport.tabIndex = 0;
        viewport.setAttribute('role', 'region');
        viewport.setAttribute('aria-label', languageLabel + ' 代码，' + lineCount + ' 行，可' +
          (wide && tall && !expanded ? '左右和上下滚动' : wide ? '左右滚动' : '上下滚动'));
      } else {
        viewport.removeAttribute('tabindex');
        viewport.removeAttribute('role');
        viewport.removeAttribute('aria-label');
      }
    }
    function scheduleUpdate() {
      if (frame === null) frame = requestAnimationFrame(update);
    }
    toggle.addEventListener('click', function() {
      expanded = !expanded;
      update();
      if (!expanded && wrapper.getBoundingClientRect().top < 80) {
        window.scrollBy({ top: wrapper.getBoundingClientRect().top - 80, behavior: 'instant' });
      }
    });
    copy.addEventListener('click', async function() {
      clearTimeout(copyTimer);
      try {
        await navigator.clipboard.writeText(source);
        copy.textContent = '已复制';
      } catch (error) {
        copy.textContent = '复制失败';
        copy.title = '请选中代码后手动复制';
      }
      copyTimer = setTimeout(function() {
        copy.textContent = '复制';
        copy.removeAttribute('title');
      }, 2000);
    });
    if (typeof ResizeObserver !== 'undefined') {
      var observer = new ResizeObserver(scheduleUpdate);
      observer.observe(pre);
      observer.observe(viewport);
    }
    window.addEventListener('resize', scheduleUpdate);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(scheduleUpdate);
    update();
  });
})();
