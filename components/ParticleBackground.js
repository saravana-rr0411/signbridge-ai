// Dynamic Roaming Particle Background Network from Stitch Design

export class ParticleBackground {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    if (!this.ctx) return;

    this.prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.animationId = null;
    this.frame = 0;
    this.maxDistance = 110;

    this.colors = [
      'rgba(15, 41, 66, ',    // Deep navy (#0f2942)
      'rgba(13, 148, 136, ',  // Crisp teal (#0d9488)
      'rgba(56, 189, 248, ',  // Soft cyan (#38bdf8)
      'rgba(100, 116, 139, '  // Slate blue (#64748b)
    ];

    this.init();
  }

  init() {
    this.resize();
    this.resizeHandler = () => this.resize();
    window.addEventListener('resize', this.resizeHandler);

    const count = 75;
    this.particles = [];

    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        radius: Math.random() * 2 + 1.8, // 1.8px - 3.8px
        vx: (Math.random() - 0.5) * 0.55,
        vy: (Math.random() - 0.5) * 0.55,
        baseAlpha: Math.random() * 0.25 + 0.18, // 0.18 - 0.43
        pulseSpeed: Math.random() * 0.02 + 0.01,
        pulseOffset: Math.random() * Math.PI * 2,
        colorBase: this.colors[Math.floor(Math.random() * this.colors.length)]
      });
    }

    this.render();
  }

  resize() {
    this.width = this.canvas.width = window.innerWidth;
    this.height = this.canvas.height = window.innerHeight;
  }

  render() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.width, this.height);
    this.frame++;

    // Update & Draw Particles
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];

      if (!this.prefersReducedMotion) {
        p.x += p.vx;
        p.y += p.vy;

        // Wrap edges smoothly
        if (p.x < -10) p.x = this.width + 10;
        else if (p.x > this.width + 10) p.x = -10;
        if (p.y < -10) p.y = this.height + 10;
        else if (p.y > this.height + 10) p.y = -10;
      }

      const alpha = p.baseAlpha + Math.sin(this.frame * p.pulseSpeed + p.pulseOffset) * 0.08;
      const boundedAlpha = Math.max(0.08, Math.min(alpha, 0.48));

      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      this.ctx.fillStyle = p.colorBase + boundedAlpha + ')';
      this.ctx.fill();

      // Connective lines to neighboring particles
      for (let j = i + 1; j < this.particles.length; j++) {
        const p2 = this.particles[j];
        const dx = p.x - p2.x;
        const dy = p.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < this.maxDistance) {
          const lineAlpha = (1 - dist / this.maxDistance) * 0.12;
          this.ctx.beginPath();
          this.ctx.moveTo(p.x, p.y);
          this.ctx.lineTo(p2.x, p2.y);
          this.ctx.strokeStyle = 'rgba(15, 41, 66, ' + lineAlpha + ')';
          this.ctx.lineWidth = 0.85;
          this.ctx.stroke();
        }
      }
    }

    if (!this.prefersReducedMotion) {
      this.animationId = requestAnimationFrame(() => this.render());
    }
  }

  destroy() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    window.removeEventListener('resize', this.resizeHandler);
  }
}
