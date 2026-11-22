(function() {
    'use strict';

    const modes = {
        dataset: {
            status: 'One dataset, three posterior weight samples.',
            top: ['Same D', 'Same D', 'Same D'],
            bottom: ['Posterior sample θᵃ', 'Posterior sample θᵇ', 'Posterior sample θᶜ'],
            description: 'Three copies of the same neural architecture share a frozen training dataset. Dashed terracotta connections change between copies, representing independent weight samples from the posterior conditioned on that dataset.'
        },
        weights: {
            status: 'One fixed network, three input samples.',
            top: ['Input xᵃ', 'Input xᵇ', 'Input xᶜ'],
            bottom: ['Same W', 'Same W', 'Same W'],
            description: 'Three copies have identical solid umber connections, representing one frozen network. Dashed terracotta input nodes represent different samples from the chosen input measure.'
        },
        probes: {
            status: 'One collection of probes, three prior weight samples.',
            top: ['Same probes', 'Same probes', 'Same probes'],
            bottom: ['Prior sample Wᵃ', 'Prior sample Wᵇ', 'Prior sample Wᶜ'],
            description: 'Three copies share the same fixed probe inputs. Dashed terracotta connections change between copies, representing different random initializations drawn from the weight prior.'
        }
    };

    function palette(figure) {
        const style = getComputedStyle(figure);
        return {
            fixed: style.getPropertyValue('--spin-ink').trim(),
            sampled: style.getPropertyValue('--spin-accent').trim(),
            ink: style.getPropertyValue('--reading-ink').trim(),
            paper: style.getPropertyValue('--reading-paper').trim(),
            rule: style.getPropertyValue('--reading-rule').trim(),
            secondary: style.getPropertyValue('--reading-secondary').trim()
        };
    }

    function prepareCanvas(canvas, ctx, width, height) {
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        canvas.style.height = height + 'px';
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);
        ctx.font = '400 15px "Alegreya", Georgia, serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
    }

    function initializeNetwork(figure) {
        const canvas = figure.querySelector('.network-canvas');
        if (!canvas) return null;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;
        const buttons = Array.from(figure.querySelectorAll('button[data-mode]'));
        const status = figure.querySelector('.network-status');
        const description = figure.querySelector('.network-description');
        let mode;

        function draw() {
            const width = figure.getBoundingClientRect().width;
            if (!width) return;
            const stacked = width < 540;
            prepareCanvas(canvas, ctx, width, stacked ? 435 : 178);
            const colors = palette(figure);
            const ys = [-32, 0, 32];
            const varyingWeights = mode !== 'weights';

            function node(x, y, varying, strength) {
                ctx.beginPath();
                ctx.arc(x, y, 7, 0, Math.PI * 2);
                ctx.fillStyle = colors.paper;
                ctx.fill();
                ctx.globalAlpha = varying ? strength : 1;
                ctx.strokeStyle = varying ? colors.sampled : colors.fixed;
                ctx.lineWidth = 1.5;
                ctx.setLineDash(varying ? [3, 2] : []);
                ctx.stroke();
                ctx.globalAlpha = 1;
                ctx.setLineDash([]);
            }

            for (let copy = 0; copy < 3; copy++) {
                const centerX = stacked ? width / 2 : (copy + 0.5) * width / 3;
                const centerY = stacked ? copy * 145 + 69 : 84;
                const inputX = centerX - 65;
                const hiddenX = centerX;
                const outputX = centerX + 65;
                ctx.fillStyle = colors.ink;
                ctx.fillText(modes[mode].top[copy], centerX, centerY - 58);

                for (let i = 0; i < 3; i++) {
                    for (let j = 0; j < 3; j++) {
                        const dx = hiddenX - inputX;
                        const dy = ys[j] - ys[i];
                        const length = Math.hypot(dx, dy);
                        const variation = varyingWeights ? ((i * 3 + j + copy * 4) % 5) / 4 : 0.55;
                        ctx.strokeStyle = varyingWeights ? colors.sampled : colors.fixed;
                        ctx.lineWidth = varyingWeights ? 0.8 + variation : 1.1;
                        ctx.globalAlpha = varyingWeights ? 0.5 + variation * 0.5 : 0.75;
                        ctx.setLineDash(varyingWeights ? [3 + variation * 2, 3] : []);
                        ctx.beginPath();
                        ctx.moveTo(inputX + dx / length * 9, centerY + ys[i] + dy / length * 9);
                        ctx.lineTo(hiddenX - dx / length * 9, centerY + ys[j] - dy / length * 9);
                        ctx.stroke();
                    }
                    const dx = outputX - hiddenX;
                    const dy = -ys[i];
                    const length = Math.hypot(dx, dy);
                    ctx.beginPath();
                    ctx.moveTo(hiddenX + dx / length * 9, centerY + ys[i] + dy / length * 9);
                    ctx.lineTo(outputX - dx / length * 9, centerY - dy / length * 9);
                    ctx.stroke();
                }
                ctx.globalAlpha = 1;
                ctx.setLineDash([]);
                ys.forEach(function(y, index) {
                    node(inputX, centerY + y, !varyingWeights, 0.6 + ((copy + index) % 3) * 0.2);
                });
                ys.forEach(function(y) { node(hiddenX, centerY + y, false, 1); });
                node(outputX, centerY, false, 1);
                ctx.fillStyle = colors.ink;
                ctx.fillText(modes[mode].bottom[copy], centerX, centerY + 58);
            }
        }

        function apply(next) {
            mode = Object.prototype.hasOwnProperty.call(modes, next) ? next : 'dataset';
            buttons.forEach(function(button) {
                button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
            });
            if (status) status.textContent = modes[mode].status;
            if (description) description.textContent = modes[mode].description;
            canvas.setAttribute('aria-label', modes[mode].status);
            draw();
        }

        buttons.forEach(function(button) {
            button.addEventListener('click', function() { apply(button.dataset.mode); });
        });
        apply(figure.dataset.mode);
        return draw;
    }

    function initialize() {
        const drawings = [];
        document.querySelectorAll('.network-figure').forEach(function(figure) {
            const draw = initializeNetwork(figure);
            if (draw) drawings.push({figure: figure, draw: draw});
        });
        function redraw() {
            drawings.forEach(function(item) { item.draw(); });
        }
        if (window.ResizeObserver) {
            const observer = new ResizeObserver(function(entries) {
                entries.forEach(function(entry) {
                    const item = drawings.find(function(drawing) { return drawing.figure === entry.target; });
                    if (item) item.draw();
                });
            });
            drawings.forEach(function(item) { observer.observe(item.figure); });
        }
        new MutationObserver(redraw).observe(document.documentElement, {
            attributes: true,
            attributeFilter: ['data-theme']
        });
        window.addEventListener('resize', redraw);
        if (document.fonts) document.fonts.ready.then(redraw);
        redraw();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize, {once: true});
    } else {
        initialize();
    }
})();

(function() {
    'use strict';

    function initializeGlossary() {
        const panel = document.getElementById('glossary-panel');
        const title = document.getElementById('glossary-title');
        const body = document.getElementById('glossary-body');
        const announcement = document.querySelector('.glossary-announcement');
        const container = document.querySelector('.container');
        if (!panel || !title || !body || !container) return;

        const hover = window.matchMedia('(hover: hover) and (pointer: fine)');
        let active = null;
        let activeKey = null;
        let pointerFocusTarget = null;
        let lastPointerType = null;
        let hoveredTerm = null;
        let pointerInPanel = false;
        let keyboardTrigger = null;
        let restoringFocus = false;
        let positionFrame = null;
        let closeTimer = null;

        function termFrom(target) {
            return target && target.closest ? target.closest('.glossary-term[data-glossary]') : null;
        }

        function position() {
            positionFrame = null;
            if (!active || panel.hidden) return;
            const viewportWidth = document.documentElement.clientWidth;
            const viewportHeight = window.innerHeight;
            const gap = 24;
            const gutter = 16;
            const edge = container.getBoundingClientRect().right;
            const available = viewportWidth - edge - gap - gutter;

            if (available >= 240) {
                panel.classList.remove('is-compact');
                panel.style.left = (edge + gap) + 'px';
                panel.style.width = Math.min(250, available) + 'px';
                const height = panel.getBoundingClientRect().height;
                const preferredTop = active.getBoundingClientRect().top - 12;
                const latestTop = Math.max(gutter, viewportHeight - height - gutter);
                panel.style.top = Math.max(gutter, Math.min(preferredTop, latestTop)) + 'px';
            } else {
                panel.classList.add('is-compact');
                panel.style.removeProperty('left');
                panel.style.removeProperty('width');
                panel.style.removeProperty('top');
            }
        }

        function schedulePosition() {
            if (!panel.hidden && positionFrame === null) {
                positionFrame = window.requestAnimationFrame(position);
            }
        }

        function cancelClose() {
            if (closeTimer !== null) {
                window.clearTimeout(closeTimer);
                closeTimer = null;
            }
        }

        function scheduleClose() {
            cancelClose();
            closeTimer = window.setTimeout(function() {
                closeTimer = null;
                const keyboardFocus = active && keyboardTrigger === active && document.activeElement === active;
                if (!pointerInPanel && hoveredTerm !== active && !keyboardFocus && !panel.contains(document.activeElement)) {
                    close(false);
                }
            }, 220);
        }

        function open(trigger) {
            const key = trigger.dataset.glossary;
            const template = document.getElementById('glossary-' + key);
            if (!template || !template.content) return;
            cancelClose();
            const wasHidden = panel.hidden;
            const changed = activeKey !== key;
            if (active && active !== trigger) active.setAttribute('aria-expanded', 'false');
            active = trigger;
            trigger.setAttribute('aria-expanded', 'true');

            if (changed || wasHidden) {
                activeKey = key;
                title.textContent = template.dataset.term || trigger.textContent.trim();
                body.replaceChildren(template.content.cloneNode(true));
            }
            panel.hidden = false;
            position();

            if (announcement && (changed || wasHidden)) {
                announcement.textContent = title.textContent + '. ' + body.textContent.trim();
            }
        }

        function close(restoreFocus) {
            cancelClose();
            if (panel.hidden) return;
            const previous = active;
            if (active) active.setAttribute('aria-expanded', 'false');
            panel.hidden = true;
            active = null;
            keyboardTrigger = null;
            if (announcement) announcement.textContent = '';
            if (restoreFocus && previous && previous.isConnected) {
                restoringFocus = true;
                previous.focus({preventScroll: true});
                restoringFocus = false;
            }
        }

        // Delegation also covers glossary buttons cloned into existing note previews.
        document.addEventListener('pointerover', function(event) {
            if (!hover.matches || event.pointerType !== 'mouse') return;
            const trigger = termFrom(event.target);
            if (trigger) {
                hoveredTerm = trigger;
                cancelClose();
                if (!event.relatedTarget || !trigger.contains(event.relatedTarget)) open(trigger);
            } else if (panel.contains(event.target)) {
                pointerInPanel = true;
                cancelClose();
            }
        });

        document.addEventListener('pointerout', function(event) {
            if (!hover.matches || event.pointerType !== 'mouse') return;
            const trigger = termFrom(event.target);
            const next = event.relatedTarget;
            if (trigger) {
                if (next && trigger.contains(next)) return;
                if (hoveredTerm === trigger) hoveredTerm = null;
            } else if (panel.contains(event.target)) {
                if (next && panel.contains(next)) return;
                pointerInPanel = false;
            } else {
                return;
            }
            if (next && (panel.contains(next) || termFrom(next))) cancelClose();
            else scheduleClose();
        });

        document.addEventListener('pointerdown', function(event) {
            lastPointerType = event.pointerType;
            keyboardTrigger = null;
            pointerFocusTarget = termFrom(event.target);
            if (!pointerFocusTarget && !panel.contains(event.target)) close(false);
        }, true);
        document.addEventListener('pointerup', function() { pointerFocusTarget = null; }, true);
        document.addEventListener('pointercancel', function() { pointerFocusTarget = null; }, true);

        document.addEventListener('focusin', function(event) {
            if (restoringFocus) return;
            const trigger = termFrom(event.target);
            if (trigger && trigger !== pointerFocusTarget) {
                keyboardTrigger = trigger;
                open(trigger);
            } else if (panel.contains(event.target)) {
                cancelClose();
            }
        });

        document.addEventListener('focusout', function(event) {
            const trigger = termFrom(event.target);
            if (!trigger && !panel.contains(event.target)) return;
            if (keyboardTrigger === trigger) keyboardTrigger = null;
            const next = event.relatedTarget;
            if (next && (panel.contains(next) || termFrom(next))) cancelClose();
            else if (!pointerInPanel) close(false);
        });

        document.addEventListener('click', function(event) {
            const trigger = termFrom(event.target);
            if (trigger) {
                const pointerType = event.pointerType || lastPointerType;
                const touch = pointerType === 'touch' || pointerType === 'pen';
                if (touch && trigger === active && !panel.hidden) close(false);
                else open(trigger);
            } else if (!panel.contains(event.target)) {
                close(false);
            }
        });

        document.addEventListener('keydown', function(event) {
            pointerFocusTarget = null;
            lastPointerType = null;
            if (event.key === 'Escape' && !panel.hidden) {
                event.preventDefault();
                close(true);
            }
        });
        window.addEventListener('scroll', schedulePosition, {passive: true});
        window.addEventListener('resize', schedulePosition);
        if (document.fonts) document.fonts.ready.then(schedulePosition);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeGlossary, {once: true});
    } else {
        initializeGlossary();
    }
})();
