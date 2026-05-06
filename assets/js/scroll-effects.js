// Scroll reveal animation with Intersection Observer
document.addEventListener('DOMContentLoaded', function() {
  // Elements to animate on scroll
  var revealTargets = [
    'section h2',
    'section .news-list',
    'section .publications',
    'section .awards-list',
    'section ol.bibliography li',
    'section p',
    'section ul:not(.news-list):not(.awards-list)',
    'section h1'
  ];

  // Add initial hidden state
  var style = document.createElement('style');
  style.textContent = '.scroll-reveal { opacity: 0; transform: translateY(24px); transition: opacity 0.6s cubic-bezier(0.4,0,0.2,1), transform 0.6s cubic-bezier(0.4,0,0.2,1); } .scroll-reveal.revealed { opacity: 1; transform: translateY(0); }';
  document.head.appendChild(style);

  // Remove old CSS animation so we control it via JS
  var oldStyle = document.createElement('style');
  oldStyle.textContent = 'section h2, section .news-list, section .publications, section .awards-list, section p, section ul:not(.news-list):not(.awards-list) { opacity: 1; transform: none; animation: none; }';
  document.head.appendChild(oldStyle);

  function initReveal() {
    var elements = [];
    revealTargets.forEach(function(selector) {
      document.querySelectorAll(selector).forEach(function(el) {
        if (!el.classList.contains('scroll-reveal')) {
          el.classList.add('scroll-reveal');
          elements.push(el);
        }
      });
    });

    if (!('IntersectionObserver' in window)) {
      // Fallback: just show everything
      elements.forEach(function(el) { el.classList.add('revealed'); });
      return;
    }

    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    elements.forEach(function(el, i) {
      el.style.transitionDelay = (i % 5) * 0.06 + 's';
      observer.observe(el);
    });
  }

  // Run after Jekyll renders content
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initReveal);
  } else {
    initReveal();
  }
  // Also re-init after any dynamic content
  setTimeout(initReveal, 500);
})();

// Reading progress bar
(function() {
  var bar = document.createElement('div');
  bar.id = 'reading-progress';
  bar.style.cssText = 'position:fixed;top:0;left:0;height:3px;width:0%;z-index:10000;background:linear-gradient(90deg,#1a3a5c 0%,#2563EB 50%,#5b8def 100%);transition:width 0.1s ease-out;pointer-events:none;';
  document.body.appendChild(bar);

  function updateProgress() {
    var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    var docHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    var progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    bar.style.width = Math.min(progress, 100) + '%';
  }

  window.addEventListener('scroll', updateProgress, { passive: true });
  window.addEventListener('resize', updateProgress, { passive: true });
  updateProgress();
})();