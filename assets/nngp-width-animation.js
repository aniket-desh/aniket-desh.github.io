/* Actual, coupled network draws. Interpolation only connects adjacent saved widths. */
(function () {
    'use strict';

    const finitePair = (point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite);

    function validateData(data) {
        if (data.schema_version !== 1 || data.samples !== 2000 || !Array.isArray(data.frames) || data.frames.length < 2) {
            throw new Error('Unsupported network animation data.');
        }
        if (!Array.isArray(data.axis_limits) || data.axis_limits.length !== 2 ||
            !data.axis_limits.every(Number.isFinite) || data.axis_limits[0] >= data.axis_limits[1] ||
            !data.ellipse || !Array.isArray(data.ellipse.points) || data.ellipse.points.length < 8 ||
            !data.ellipse.points.every(finitePair)) {
            throw new Error('Invalid plot geometry.');
        }
        let previous = 0;
        for (const frame of data.frames) {
            if (!Number.isInteger(frame.width) || frame.width <= previous ||
                !Array.isArray(frame.points) || frame.points.length !== data.samples ||
                !frame.points.every(finitePair)) {
                throw new Error('Invalid network samples.');
            }
            previous = frame.width;
        }
        return data;
    }

    async function initialize(root) {
        const canvas = root.querySelector('[data-nngp-canvas]');
        const controls = root.querySelector('[data-nngp-controls]');
        const play = root.querySelector('[data-nngp-play]');
        const range = root.querySelector('[data-nngp-range]');
        const readout = root.querySelector('[data-nngp-readout]');
        const status = root.querySelector('[data-nngp-status]');
        const announcement = root.querySelector('[data-nngp-announcement]');
        const fallback = root.querySelector('[data-nngp-fallback]');
        const key = root.querySelector('[data-nngp-key]');
        if (![canvas, controls, play, range, readout, status, announcement, fallback, key].every(Boolean)) return;
        const context = canvas.getContext('2d');
        if (!context) return;

        let data;
        const abort = new AbortController();
        const timeout = window.setTimeout(() => abort.abort(), 15000);
        try {
            const response = await fetch(root.dataset.src, { signal: abort.signal });
            if (!response.ok) throw new Error('Network samples unavailable.');
            data = validateData(await response.json());
        } catch (error) {
            status.hidden = false;
            status.textContent = 'The animation could not load. The static comparison is shown above.';
            return;
        } finally {
            window.clearTimeout(timeout);
        }

        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        const frames = data.frames;
        const limits = data.axis_limits;
        const duration = 520;
        let index = 0;
        let progress = 0;
        let elapsed = 0;
        let lastTime = null;
        let animation = 0;
        let playing = false;
        let visible = false;
        let autoplayAttempted = false;
        let width = 0;
        let height = 0;
        let palette;

        function colors() {
            const style = getComputedStyle(root);
            palette = {
                point: style.getPropertyValue('--nngp-point').trim() || '#a65332',
                ink: style.getPropertyValue('--nngp-ink').trim() || '#62472f',
                rule: style.getPropertyValue('--nngp-rule').trim() || '#dedbd5',
                secondary: style.getPropertyValue('--reading-secondary').trim() || '#62605c'
            };
        }

        function announce(message) {
            announcement.textContent = message;
        }

        function updateControls() {
            range.value = String(index);
            range.setAttribute('aria-valuetext', frames[index].width + ' neurons');
            readout.textContent = 'N = ' + frames[index].width.toLocaleString();
            play.textContent = playing ? 'Pause' : (index === frames.length - 1 ? 'Replay' : 'Play');
            play.setAttribute('aria-label', playing ? 'Pause width animation' : 'Play width animation');
            canvas.setAttribute('aria-label', 'Two thousand network output pairs at width ' + frames[index].width +
                ', compared with the Gaussian process 95 percent ellipse.');
        }

        function draw() {
            if (!width || !height) return;
            const font = width < 440 ? 13 : 15;
            const side = Math.min(width - 60, height - 58);
            const left = (width - side) / 2 + 6;
            const top = 10;
            const scale = side / (limits[1] - limits[0]);
            const px = (value) => left + (value - limits[0]) * scale;
            const py = (value) => top + (limits[1] - value) * scale;
            context.clearRect(0, 0, width, height);
            context.font = font + 'px Alegreya, Georgia, serif';
            context.strokeStyle = palette.rule;
            context.lineWidth = 0.7;
            context.beginPath();
            context.moveTo(px(0), top);
            context.lineTo(px(0), top + side);
            context.moveTo(left, py(0));
            context.lineTo(left + side, py(0));
            context.stroke();

            context.save();
            context.beginPath();
            context.rect(left, top, side, side);
            context.clip();
            const next = frames[Math.min(index + 1, frames.length - 1)];
            const blend = progress * progress * (3 - 2 * progress);
            let outside = 0;
            context.fillStyle = palette.point;
            context.globalAlpha = 0.2;
            const radius = width < 440 ? 1.15 : 1.5;
            for (let point = 0; point < data.samples; point++) {
                const a = frames[index].points[point];
                const b = next.points[point];
                const x = a[0] + blend * (b[0] - a[0]);
                const y = a[1] + blend * (b[1] - a[1]);
                if (x < limits[0] || x > limits[1] || y < limits[0] || y > limits[1]) {
                    outside++;
                    continue;
                }
                context.beginPath();
                context.arc(px(x), py(y), radius, 0, Math.PI * 2);
                context.fill();
            }
            context.globalAlpha = 1;
            context.strokeStyle = palette.ink;
            context.lineWidth = 1.5;
            context.setLineDash([6, 5]);
            context.beginPath();
            data.ellipse.points.forEach((point, i) => {
                if (i === 0) context.moveTo(px(point[0]), py(point[1]));
                else context.lineTo(px(point[0]), py(point[1]));
            });
            context.closePath();
            context.stroke();
            context.restore();

            context.strokeStyle = palette.rule;
            context.lineWidth = 0.8;
            context.strokeRect(left, top, side, side);
            context.fillStyle = palette.secondary;
            context.textAlign = 'center';
            context.textBaseline = 'top';
            for (const tick of [-4, -2, 0, 2, 4]) {
                context.fillText(String(tick).replace('-', '−'), px(tick), top + side + 7);
            }
            context.textAlign = 'right';
            context.textBaseline = 'middle';
            for (const tick of [-4, -2, 0, 2, 4]) {
                context.fillText(String(tick).replace('-', '−'), left - 8, py(tick));
            }
            context.fillStyle = palette.ink;
            context.textAlign = 'center';
            context.textBaseline = 'bottom';
            context.fillText('Output at input 1', left + side / 2, height - 1);
            context.save();
            context.translate(left - 35, top + side / 2);
            context.rotate(-Math.PI / 2);
            context.fillText('Output at input 2', 0, 0);
            context.restore();

            const movement = progress > 0 ? 'Moving to N = ' + next.width.toLocaleString() + ' · ' : '';
            const endpoint = index === frames.length - 1 ? ' · finite width endpoint' : '';
            status.textContent = movement + data.samples.toLocaleString() + ' networks · ' + outside +
                (outside === 1 ? ' point outside view' : ' points outside view') + endpoint;
        }

        function resize() {
            width = canvas.getBoundingClientRect().width;
            height = width / 1.12;
            const ratio = Math.min(window.devicePixelRatio || 1, 3);
            canvas.width = Math.round(width * ratio);
            canvas.height = Math.round(height * ratio);
            context.setTransform(ratio, 0, 0, ratio, 0, 0);
            draw();
        }

        function cancelFrame() {
            window.cancelAnimationFrame(animation);
            animation = 0;
            lastTime = null;
        }

        function pause() {
            playing = false;
            cancelFrame();
            // A paused cloud always corresponds to an actual saved width.
            elapsed = 0;
            progress = 0;
            updateControls();
            draw();
        }

        function step(time) {
            animation = 0;
            if (!playing || !visible || document.hidden) {
                lastTime = null;
                return;
            }
            if (lastTime !== null) elapsed += Math.min(time - lastTime, 100);
            lastTime = time;
            if (elapsed >= duration) {
                elapsed = 0;
                index++;
                updateControls();
            }
            if (index === frames.length - 1) {
                pause();
                announce('Animation complete at finite width ' + frames[index].width + '.');
                return;
            }
            progress = reducedMotion.matches ? 0 : elapsed / duration;
            draw();
            animation = window.requestAnimationFrame(step);
        }

        function start() {
            if (index === frames.length - 1) {
                index = 0;
                elapsed = 0;
                progress = 0;
            }
            playing = true;
            updateControls();
            draw();
            announce('Playing network widths from ' + frames[index].width + ' to ' + frames[frames.length - 1].width + '.');
            if (visible && !document.hidden && !animation) animation = window.requestAnimationFrame(step);
        }

        play.addEventListener('click', () => {
            autoplayAttempted = true;
            if (playing) {
                pause();
                announce('Paused at width ' + frames[index].width + '.');
            } else start();
        });
        range.min = '0';
        range.max = String(frames.length - 1);
        range.step = '1';
        range.addEventListener('input', () => {
            autoplayAttempted = true;
            index = Math.max(0, Math.min(frames.length - 1, Number(range.value)));
            pause();
        });
        range.addEventListener('change', () => announce('Width ' + frames[index].width + ' selected.'));

        colors();
        canvas.hidden = false;
        controls.hidden = false;
        status.hidden = false;
        key.hidden = false;
        fallback.hidden = true;
        updateControls();
        resize();
        if ('ResizeObserver' in window) new ResizeObserver(resize).observe(canvas);
        else window.addEventListener('resize', resize);
        new MutationObserver(() => { colors(); draw(); }).observe(document.documentElement, {
            attributes: true, attributeFilter: ['data-theme']
        });
        if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);

        function onVisibility(isVisible) {
            visible = isVisible;
            if (!visible) cancelFrame();
            else if (!autoplayAttempted) {
                autoplayAttempted = true;
                if (!reducedMotion.matches) start();
            } else if (playing && !document.hidden && !animation) animation = window.requestAnimationFrame(step);
        }
        if ('IntersectionObserver' in window) {
            new IntersectionObserver((entries) => onVisibility(entries[0].isIntersecting && entries[0].intersectionRatio >= 0.2), {
                threshold: 0.2
            }).observe(canvas);
        } else onVisibility(true);
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) cancelFrame();
            else if (playing && visible && !animation) animation = window.requestAnimationFrame(step);
        });
        const motionChanged = () => { if (reducedMotion.matches) pause(); };
        if (reducedMotion.addEventListener) reducedMotion.addEventListener('change', motionChanged);
        else reducedMotion.addListener(motionChanged);
    }

    document.querySelectorAll('[data-nngp-width]').forEach(initialize);
})();
