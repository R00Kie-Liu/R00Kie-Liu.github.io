(function() {
  var filter = document.getElementById('tagFilter');
  var search = document.getElementById('blogSearch');
  var noResults = document.getElementById('blogNoResults');
  var count = document.getElementById('blogResultCount');
  var more = document.getElementById('moreTopics');
  var moreLabel = document.getElementById('moreTopicsLabel');
  var buttons = filter ? filter.querySelectorAll('[data-tag]') : [];
  var cards = document.querySelectorAll('.post-card');
  var years = document.querySelectorAll('.blog-year');
  var resetButtons = document.querySelectorAll('[data-reset-filters]');
  var activeTag = 'all';

  function applyFilters() {
    var query = search.value.trim().toLowerCase();
    var visibleCount = 0;
    cards.forEach(function(card) {
      var tags = card.getAttribute('data-tags').split(',');
      var matches = (activeTag === 'all' || tags.indexOf(activeTag) >= 0) &&
        (!query || card.getAttribute('data-search').indexOf(query) >= 0);
      card.hidden = !matches;
      if (matches) visibleCount++;
    });
    years.forEach(function(year) {
      year.hidden = !year.querySelector('.post-card:not([hidden])');
    });
    noResults.hidden = visibleCount > 0 || cards.length === 0;
    count.textContent = visibleCount + (visibleCount === 1 ? ' note' : ' notes');
    resetButtons.forEach(function(button) {
      button.hidden = activeTag === 'all' && !query;
    });
    buttons.forEach(function(button) {
      var selected = button.getAttribute('data-tag') === activeTag;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    if (moreLabel) {
      var extraSelected = more.querySelector('[data-tag][aria-pressed="true"]');
      moreLabel.textContent = extraSelected ? 'Topic: ' + activeTag : 'More topics';
    }
  }

  buttons.forEach(function(button) {
    button.addEventListener('click', function() {
      activeTag = button.getAttribute('data-tag');
      if (more && more.contains(button)) {
        more.open = false;
        moreLabel.focus();
      }
      applyFilters();
    });
  });
  resetButtons.forEach(function(button) {
    button.addEventListener('click', function() {
      activeTag = 'all';
      search.value = '';
      if (more) more.open = false;
      applyFilters();
      search.focus();
    });
  });
  document.addEventListener('click', function(event) {
    if (more && !more.contains(event.target)) more.open = false;
  });
  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && more && more.open) {
      more.open = false;
      moreLabel.focus();
    }
  });
  search.addEventListener('input', applyFilters);
  applyFilters();
})();
