(function() {
  var content = document.querySelector('.post-content');
  if (!content) return;

  var tables = content.querySelectorAll('table');
  tables.forEach(function(table, index) {
    // Leave explicitly managed or nested tables alone.
    if (table.closest('.post-table') || table.parentElement.closest('table')) return;

    var wrapper = document.createElement('div');
    wrapper.className = 'post-table';
    var toolbar = document.createElement('div');
    toolbar.className = 'post-table-toolbar';
    toolbar.hidden = true;
    var info = document.createElement('span');
    info.className = 'post-table-info';
    var toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'post-table-toggle';
    toggle.hidden = true;
    var viewport = document.createElement('div');
    viewport.className = 'post-table-scroll';
    var id = 'post-table-' + (index + 1);
    while (document.getElementById(id)) id += '-scroll';
    viewport.id = id;
    toggle.setAttribute('aria-controls', id);
    toolbar.appendChild(info);
    toolbar.appendChild(toggle);
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(toolbar);
    wrapper.appendChild(viewport);
    viewport.appendChild(table);

    var expanded = false;
    var drag = null;
    var dragged = false;
    var frame = null;
    var caption = table.caption && table.caption.textContent.trim();
    var label = caption || '表格 ' + (index + 1);

    function update() {
      frame = null;
      var rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      var limit = Math.floor(Math.min(window.innerHeight * 0.65, rem * 36));
      viewport.style.setProperty('--post-table-height', limit + 'px');
      var tall = table.getBoundingClientRect().height > limit + 1;
      if (!tall) expanded = false;
      wrapper.classList.toggle('is-expanded', expanded);
      var wide = viewport.scrollWidth > viewport.clientWidth + 1;
      var scrollable = wide || (tall && !expanded);
      toolbar.hidden = !wide && !tall;
      toggle.hidden = !tall;
      toggle.textContent = expanded ? '收起表格' : '展开表格';
      toggle.setAttribute('aria-expanded', String(expanded));
      var rows = table.tBodies.length ? Array.prototype.reduce.call(table.tBodies, function(n, body) {
        return n + body.rows.length;
      }, 0) : table.rows.length;
      var columns = table.rows.length ? Array.prototype.reduce.call(table.rows[0].cells, function(n, cell) {
        return n + cell.colSpan;
      }, 0) : 0;
      var directions = [];
      if (wide) directions.push('左右滚动');
      if (tall && !expanded) directions.push('上下滚动');
      info.textContent = rows + ' 行 × ' + columns + ' 列' +
        (directions.length ? ' · 可' + directions.join(' / ') : '');
      wrapper.classList.toggle('is-scrollable', scrollable);
      if (scrollable) {
        viewport.tabIndex = 0;
        viewport.setAttribute('role', 'region');
        viewport.setAttribute('aria-label', label + '，' + info.textContent);
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
      // Keep the control visible when collapsing a table from far down the page.
      if (!expanded && wrapper.getBoundingClientRect().top < 60) {
        window.scrollBy(0, wrapper.getBoundingClientRect().top - 60);
      }
      update();
    });

    // Touch and trackpad scrolling stay native. Mouse dragging is an additional
    // option; links, controls, and Shift-drag text selection remain available.
    viewport.addEventListener('pointerdown', function(event) {
      if (event.pointerType !== 'mouse' || event.button !== 0 || event.shiftKey ||
          event.target.closest('a, button, input, select, textarea, label, [contenteditable]')) return;
      if (viewport.scrollWidth <= viewport.clientWidth + 1 &&
          viewport.scrollHeight <= viewport.clientHeight + 1) return;
      dragged = false;
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY,
        left: viewport.scrollLeft, top: viewport.scrollTop };
    });
    viewport.addEventListener('pointermove', function(event) {
      if (!drag || event.pointerId !== drag.id) return;
      var dx = event.clientX - drag.x;
      var dy = event.clientY - drag.y;
      if (!dragged && Math.hypot(dx, dy) < 6) return;
      if (!dragged) {
        dragged = true;
        viewport.setPointerCapture(event.pointerId);
        wrapper.classList.add('is-dragging');
      }
      event.preventDefault();
      var selection = window.getSelection();
      if (selection) selection.removeAllRanges();
      viewport.scrollLeft = drag.left - dx;
      viewport.scrollTop = drag.top - dy;
    });
    function endDrag() {
      if (drag && viewport.hasPointerCapture(drag.id)) viewport.releasePointerCapture(drag.id);
      drag = null;
      wrapper.classList.remove('is-dragging');
      // A synthesized click follows pointerup; do not activate anything after a pan.
      setTimeout(function() { dragged = false; }, 0);
    }
    viewport.addEventListener('pointerup', endDrag);
    viewport.addEventListener('pointercancel', endDrag);
    viewport.addEventListener('lostpointercapture', endDrag);
    viewport.addEventListener('pointerleave', function() {
      if (!dragged) drag = null;
    });
    viewport.addEventListener('click', function(event) {
      if (dragged) {
        event.preventDefault();
        event.stopPropagation();
      }
    }, true);

    if (typeof ResizeObserver !== 'undefined') {
      var observer = new ResizeObserver(scheduleUpdate);
      observer.observe(table);
      observer.observe(viewport);
    }
    window.addEventListener('resize', scheduleUpdate);
    window.addEventListener('load', scheduleUpdate);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(scheduleUpdate);
    update();
  });
})();
