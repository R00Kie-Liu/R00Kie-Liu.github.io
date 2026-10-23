(function() {
  var button = document.getElementById('navMenuBtn');
  var drawer = document.getElementById('navDrawer');
  var overlay = document.getElementById('navOverlay');
  var closeButton = document.getElementById('navDrawerClose');
  var mobile = window.matchMedia('(max-width: 960px)');
  var previousOverflow = '';

  function isOpen() {
    return button.getAttribute('aria-expanded') === 'true';
  }

  function closeDrawer(restoreFocus) {
    if (!isOpen()) return;
    drawer.classList.remove('active');
    overlay.classList.remove('active');
    drawer.setAttribute('aria-hidden', 'true');
    drawer.setAttribute('inert', '');
    button.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = previousOverflow;
    if (restoreFocus) button.focus();
  }

  button.addEventListener('click', function() {
    if (!mobile.matches) return;
    previousOverflow = document.body.style.overflow;
    drawer.removeAttribute('inert');
    drawer.setAttribute('aria-hidden', 'false');
    drawer.classList.add('active');
    overlay.classList.add('active');
    button.setAttribute('aria-expanded', 'true');
    document.body.style.overflow = 'hidden';
    requestAnimationFrame(function() {
      if (isOpen()) closeButton.focus();
    });
  });

  closeButton.addEventListener('click', function() { closeDrawer(true); });
  drawer.addEventListener('transitionend', function(event) {
    if (event.propertyName === 'right' && isOpen() && !drawer.contains(document.activeElement)) {
      closeButton.focus();
    }
  });
  overlay.addEventListener('click', function() { closeDrawer(true); });
  drawer.querySelectorAll('a').forEach(function(link) {
    link.addEventListener('click', function() { closeDrawer(true); });
  });
  document.addEventListener('keydown', function(event) {
    if (!isOpen()) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      closeDrawer(true);
    } else if (event.key === 'Tab') {
      var controls = drawer.querySelectorAll('button, a[href]');
      var first = controls[0];
      var last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  mobile.addEventListener('change', function() {
    if (!mobile.matches) closeDrawer(false);
  });
})();
