(function() {
  var postContent = document.querySelector('.post-content');
  if (!postContent) return;

  var images = Array.prototype.slice.call(postContent.querySelectorAll('img'));
  if (!images.length) return;

  var lightbox = document.createElement('div');
  lightbox.className = 'post-image-lightbox';
  lightbox.setAttribute('role', 'dialog');
  lightbox.setAttribute('aria-modal', 'true');
  lightbox.setAttribute('aria-hidden', 'true');

  var closeButton = document.createElement('button');
  closeButton.type = 'button';
  closeButton.className = 'post-image-lightbox-close';
  closeButton.setAttribute('aria-label', '关闭图片预览');
  closeButton.innerHTML = '<i class="fas fa-xmark" aria-hidden="true"></i>';

  var previewImage = document.createElement('img');
  previewImage.className = 'post-image-lightbox-img';
  previewImage.alt = '';

  var caption = document.createElement('div');
  caption.className = 'post-image-lightbox-caption';

  lightbox.appendChild(closeButton);
  lightbox.appendChild(previewImage);
  lightbox.appendChild(caption);
  document.body.appendChild(lightbox);

  var activeButton = null;

  function openLightbox(img, button) {
    var captionText = img.getAttribute('alt') || '';
    activeButton = button;
    previewImage.src = img.currentSrc || img.src;
    previewImage.alt = captionText || '放大图片';
    caption.textContent = captionText;
    caption.hidden = !captionText;
    lightbox.classList.add('open');
    lightbox.setAttribute('aria-hidden', 'false');
    document.body.classList.add('image-lightbox-open');
    closeButton.focus();
  }

  function closeLightbox() {
    lightbox.classList.remove('open');
    lightbox.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('image-lightbox-open');
    previewImage.removeAttribute('src');
    if (activeButton) activeButton.focus();
    activeButton = null;
  }

  images.forEach(function(img) {
    if (img.closest('a') || img.closest('.post-image-frame') || img.dataset.noLightbox === 'true') return;

    var frame = document.createElement('span');
    frame.className = 'post-image-frame';
    img.parentNode.insertBefore(frame, img);
    frame.appendChild(img);

    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'post-image-expand';
    button.setAttribute('aria-label', '放大图片');
    button.setAttribute('title', '放大图片');
    button.innerHTML = '<i class="fas fa-expand" aria-hidden="true"></i>';
    button.addEventListener('click', function() {
      openLightbox(img, button);
    });
    frame.appendChild(button);
  });

  closeButton.addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', function(event) {
    if (event.target === lightbox) closeLightbox();
  });

  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && lightbox.classList.contains('open')) {
      closeLightbox();
    }
  });
})();
