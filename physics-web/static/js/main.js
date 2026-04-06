/**
 * Main JS for 부산대학교 물리학과 웹페이지
 */

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const id = a.getAttribute('href').slice(1);
    const el = document.getElementById(id);
    if (el) {
      e.preventDefault();
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Active nav link highlight (for same-page sections)
const sections = document.querySelectorAll('section[id]');
if (sections.length) {
  const navLinks = document.querySelectorAll('nav a');
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(link => {
          if (link.getAttribute('href') === '#' + entry.target.id) {
            link.classList.add('text-pnu-gold');
          } else {
            link.classList.remove('text-pnu-gold');
          }
        });
      }
    });
  }, { threshold: 0.5 });
  sections.forEach(s => io.observe(s));
}

// Faculty filter (for faculty page)
const filterBtns = document.querySelectorAll('[data-filter]');
const facultyCards = document.querySelectorAll('[data-specialty]');

filterBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const filter = btn.dataset.filter;
    filterBtns.forEach(b => {
      b.classList.remove('bg-pnu-blue', 'text-white');
      b.classList.add('bg-white/5', 'text-gray-400');
    });
    btn.classList.add('bg-pnu-blue', 'text-white');
    btn.classList.remove('bg-white/5', 'text-gray-400');

    facultyCards.forEach(card => {
      if (filter === 'all' || card.dataset.specialty === filter) {
        card.style.display = '';
        card.classList.add('animate-fadeIn');
      } else {
        card.style.display = 'none';
      }
    });
  });
});

// Faculty search
const searchInput = document.getElementById('faculty-search');
if (searchInput) {
  searchInput.addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    facultyCards.forEach(card => {
      const text = card.textContent.toLowerCase();
      card.style.display = text.includes(q) ? '' : 'none';
    });
  });
}

// Contact form
const contactForm = document.getElementById('contact-form');
if (contactForm) {
  contactForm.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = contactForm.querySelector('[type="submit"]');
    const orig = btn.textContent;
    btn.textContent = '전송 중...';
    btn.disabled = true;
    await new Promise(r => setTimeout(r, 1000));
    btn.textContent = '✓ 전송 완료!';
    btn.classList.add('bg-green-600');
    setTimeout(() => {
      btn.textContent = orig;
      btn.disabled = false;
      btn.classList.remove('bg-green-600');
      contactForm.reset();
    }, 3000);
  });
}

// Research accordion
const accordionBtns = document.querySelectorAll('[data-accordion]');
accordionBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const id = btn.dataset.accordion;
    const panel = document.getElementById('panel-' + id);
    const icon = btn.querySelector('[data-icon]');
    const isOpen = panel.style.maxHeight;

    // Close all
    document.querySelectorAll('.accordion-panel').forEach(p => {
      p.style.maxHeight = '';
      p.style.opacity = '0';
    });
    document.querySelectorAll('[data-icon]').forEach(i => {
      i.style.transform = 'rotate(0deg)';
    });

    // Toggle clicked
    if (!isOpen) {
      panel.style.maxHeight = panel.scrollHeight + 'px';
      panel.style.opacity = '1';
      if (icon) icon.style.transform = 'rotate(180deg)';
    }
  });
});
