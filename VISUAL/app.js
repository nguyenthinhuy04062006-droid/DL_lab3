// Lightbox Modal for Fullscreen Image Preview
function openModal(src) {
  const modal = document.getElementById('img-modal');
  const target = document.getElementById('modal-target');
  target.src = src;
  modal.classList.add('active');
}

function closeModal() {
  const modal = document.getElementById('img-modal');
  modal.classList.remove('active');
}

// ESC Key to close modal
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeModal();
  }
});
