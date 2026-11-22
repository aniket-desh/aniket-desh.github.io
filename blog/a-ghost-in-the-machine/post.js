(() => {
    const sections = Array.from(document.querySelectorAll('.section-title[id]'));
    const tocLinks = Array.from(document.querySelectorAll('.toc-sidebar a'));
    const byId = new Map(tocLinks.map((link) => [link.getAttribute('href').slice(1), link]));

    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
            if (!visible.length) return;
            tocLinks.forEach((link) => link.classList.remove('active'));
            const active = byId.get(visible[0].target.id);
            if (active) active.classList.add('active');
        }, { rootMargin: '-18% 0px -68% 0px', threshold: 0 });
        sections.forEach((section) => observer.observe(section));
    }

    const video = document.getElementById('project-video');
    const toggle = document.getElementById('project-toggle');
    const playIcon = document.getElementById('project-play-icon');
    const pauseIcon = document.getElementById('project-pause-icon');
    const toggleLabel = document.getElementById('project-toggle-label');
    const note = document.getElementById('project-motion-note');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

    const updateButton = () => {
        const paused = video.paused;
        playIcon.hidden = !paused;
        pauseIcon.hidden = paused;
        toggleLabel.textContent = paused ? 'Play animation' : 'Pause animation';
        toggle.setAttribute('aria-label', paused ? 'Play parameter sweep animation' : 'Pause parameter sweep animation');
    };

    const applyMotionPreference = () => {
        if (reducedMotion.matches) {
            video.pause();
            note.textContent = 'Animation paused because reduced motion is enabled.';
        } else {
            note.textContent = 'The movie loops through the sampled rank-one strengths.';
            video.play().catch(() => updateButton());
        }
        updateButton();
    };

    toggle.addEventListener('click', async () => {
        if (video.paused) {
            try {
                await video.play();
            } catch (_) {
                // The poster remains usable when a browser blocks playback.
            }
        } else {
            video.pause();
        }
        updateButton();
    });
    video.addEventListener('play', updateButton);
    video.addEventListener('pause', updateButton);
    reducedMotion.addEventListener?.('change', applyMotionPreference);
    applyMotionPreference();
})();
