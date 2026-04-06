/**
 * Particle Background Animation for Hero Section
 * Physics-inspired: particles connected by lines, react to mouse
 */
(function () {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let W, H, particles = [], mouse = { x: null, y: null };

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x    = Math.random() * W;
      this.y    = Math.random() * H;
      this.vx   = (Math.random() - 0.5) * 0.6;
      this.vy   = (Math.random() - 0.5) * 0.6;
      this.r    = Math.random() * 2 + 1;
      this.life = Math.random();
    }
    update() {
      // Mouse repulsion
      if (mouse.x !== null) {
        const dx = this.x - mouse.x;
        const dy = this.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          const force = (120 - dist) / 120;
          this.vx += (dx / dist) * force * 0.5;
          this.vy += (dy / dist) * force * 0.5;
        }
      }
      // Speed limit
      const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
      if (speed > 2) { this.vx *= 0.95; this.vy *= 0.95; }

      this.x += this.vx;
      this.y += this.vy;

      // Wrap around
      if (this.x < 0) this.x = W;
      if (this.x > W) this.x = 0;
      if (this.y < 0) this.y = H;
      if (this.y > H) this.y = 0;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(100, 160, 255, ${0.4 + this.life * 0.4})`;
      ctx.fill();
    }
  }

  function init() {
    resize();
    const count = Math.floor((W * H) / 12000);
    particles = Array.from({ length: Math.min(count, 90) }, () => new Particle());
  }

  function drawConnections() {
    const maxDist = 150;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < maxDist) {
          const alpha = (1 - d / maxDist) * 0.25;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 102, 204, ${alpha})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    drawConnections();
    requestAnimationFrame(loop);
  }

  window.addEventListener('resize', init);
  window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  window.addEventListener('mouseleave', () => { mouse.x = null; mouse.y = null; });

  init();
  loop();
})();
