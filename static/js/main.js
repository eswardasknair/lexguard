/* ============================================================
   LexGuard — Main JS
   Particle system, scroll-reveal, counter animations, drag-drop
   ============================================================ */

/* ── 1. Scroll-Reveal ──────────────────────────────────────── */
(function initReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.12 });

    document.querySelectorAll('.reveal, .reveal-left, .reveal-right')
        .forEach(el => observer.observe(el));
})();

/* ── 2. Animated Counter ───────────────────────────────────── */
(function initCounters() {
    const counters = document.querySelectorAll('[data-count]');
    if (!counters.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseInt(el.getAttribute('data-count'), 10);
            const suffix = el.getAttribute('data-suffix') || '';
            const duration = 1800;
            const start = performance.now();

            function tick(now) {
                const elapsed = now - start;
                const progress = Math.min(elapsed / duration, 1);
                // Ease-out cubic
                const eased = 1 - Math.pow(1 - progress, 3);
                el.textContent = Math.round(eased * target) + suffix;
                if (progress < 1) requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
            observer.unobserve(el);
        });
    }, { threshold: 0.3 });

    counters.forEach(el => observer.observe(el));
})();

/* ── 3. Particle Canvas (Hero) ─────────────────────────────── */
(function initParticles() {
    const canvas = document.getElementById('particles-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let W, H, particles = [], animFrame;

    function resize() {
        W = canvas.width = canvas.offsetWidth;
        H = canvas.height = canvas.offsetHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    const COLORS = ['rgba(79,142,247,', 'rgba(124,58,237,', 'rgba(6,182,212,'];
    const COUNT = window.innerWidth < 600 ? 40 : 80;

    function rand(a, b) { return Math.random() * (b - a) + a; }

    for (let i = 0; i < COUNT; i++) {
        particles.push({
            x: rand(0, W), y: rand(0, H),
            r: rand(1, 3),
            vx: rand(-0.3, 0.3), vy: rand(-0.4, -0.1),
            alpha: rand(0.3, 0.8),
            color: COLORS[Math.floor(Math.random() * COLORS.length)]
        });
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = p.color + p.alpha + ')';
            ctx.fill();

            p.x += p.vx; p.y += p.vy;
            p.alpha -= 0.0015;

            if (p.y < -5 || p.alpha <= 0) {
                p.x = rand(0, W); p.y = H + 5;
                p.alpha = rand(0.3, 0.8);
                p.vx = rand(-0.3, 0.3);
                p.vy = rand(-0.4, -0.1);
            }
        });

        // Draw faint connection lines
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 90) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(79,142,247,${0.07 * (1 - dist / 90)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        animFrame = requestAnimationFrame(draw);
    }
    draw();
})();

/* ── 4. Navbar active link highlight ───────────────────────── */
(function highlightNav() {
    const path = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === path) link.classList.add('active');
    });
})();

/* ── 5. Upload: showFileName (global, called from inline HTML) */
function showFileName() {
    const input = document.getElementById('fileInput');
    const preview = document.getElementById('filePreview');
    const nameDisp = document.getElementById('fileNameDisplay');
    const dropArea = document.getElementById('dropArea');
    const uploadIdle = document.getElementById('uploadIdle');
    const uploadSuccess = document.getElementById('uploadSuccess');

    if (input && input.files && input.files[0]) {
        if (nameDisp) nameDisp.textContent = input.files[0].name;
        if (preview) preview.classList.remove('d-none');
        if (uploadIdle) uploadIdle.classList.add('d-none');
        if (uploadSuccess) uploadSuccess.classList.remove('d-none');
        if (dropArea) {
            dropArea.classList.add('upload-success');
            dropArea.classList.remove('upload-zone');
        }
    }
}

/* ── 6. Smooth page-load fade-in ───────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.4s ease';
    requestAnimationFrame(() => { document.body.style.opacity = '1'; });

    // Re-trigger reveal for anything already in viewport on load
    setTimeout(() => {
        document.querySelectorAll('.reveal, .reveal-left, .reveal-right')
            .forEach(el => {
                const rect = el.getBoundingClientRect();
                if (rect.top < window.innerHeight) el.classList.add('visible');
            });
    }, 100);
});
