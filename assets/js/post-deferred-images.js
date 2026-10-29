(function() {
  var images = document.querySelectorAll('.post-content img[data-deferred-src]');
  function load(image) {
    image.src = image.dataset.deferredSrc;
    image.removeAttribute('data-deferred-src');
  }
  if (typeof IntersectionObserver === 'undefined') {
    images.forEach(load);
    return;
  }
  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (!entry.isIntersecting) return;
      observer.unobserve(entry.target);
      load(entry.target);
    });
  }, { rootMargin: '300px 0px' });
  images.forEach(function(image) { observer.observe(image); });
})();
