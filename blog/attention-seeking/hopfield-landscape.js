(() => {
    "use strict";

    const COLORS = Object.freeze({
        white: "#ffffff",
        ink: "#2e3440",
        axis: "#4c566a",
        grid: "#d8dee9",
        snow2: "#e5e9f0",
        snow3: "#eceff4",
        blue: "#5e81ac",
        blueMid: "#81a1c1",
        cyan: "#88c0d0",
        teal: "#8fbcbb",
        yellow: "#ebcb8b",
        red: "#bf616a",
    });

    const SVG_NS = "http://www.w3.org/2000/svg";
    const WIDTH = 720;
    const HEIGHT = 430;
    const MARGIN = { top: 10, right: 14, bottom: 10, left: 14 };
    const X_MIN = -2.2;
    const X_MAX = 2.2;
    const Y_MIN = -2.0;
    const Y_MAX = 2.0;
    const INITIAL_CUE = Object.freeze([0.6, 0.15]);
    const PATTERN_RADIUS = 1.45;
    const PATTERN_ANGLES = Object.freeze([0.35, -0.35, Math.PI]);
    const PATTERNS = Object.freeze(
        PATTERN_ANGLES.map((angle) =>
            Object.freeze([
                PATTERN_RADIUS * Math.cos(angle),
                PATTERN_RADIUS * Math.sin(angle),
            ])
        )
    );

    function init() {
        const root = document.getElementById("hopfield-landscape-root");
        if (!root) return;

        const betaInput = root.querySelector("#hopfield-beta");
        const betaValue = root.querySelector("#hopfield-beta-value");
        const playButton = root.querySelector("#hopfield-play");
        const stepButton = root.querySelector("#hopfield-step");
        const resetButton = root.querySelector("#hopfield-reset");
        const regimeOutput = root.querySelector("#hopfield-regime");
        const svg = root.querySelector("#hopfield-landscape-svg");
        if (
            !betaInput ||
            !betaValue ||
            !playButton ||
            !stepButton ||
            !resetButton ||
            !regimeOutput ||
            !svg
        ) {
            return;
        }

        const plotWidth = WIDTH - MARGIN.left - MARGIN.right;
        const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
        const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
        let beta = sanitizeBeta(betaInput.value);
        let cue = [...INITIAL_CUE];
        let trajectory = [cue.slice()];
        let playing = false;
        let playToken = 0;
        let dragging = false;

        svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
        svg.setAttribute("role", "img");
        svg.setAttribute(
            "aria-label",
            "Modern Hopfield energy landscape with three stored patterns and an interactive retrieval cue"
        );
        svg.style.width = "100%";
        svg.style.height = "auto";
        svg.style.display = "block";
        svg.style.background = COLORS.white;
        svg.style.touchAction = "none";

        function sanitizeBeta(value) {
            const parsed = Number(value);
            return Number.isFinite(parsed) ? Math.max(0.05, parsed) : 1;
        }

        function xToSvg(x) {
            return MARGIN.left + ((x - X_MIN) / (X_MAX - X_MIN)) * plotWidth;
        }

        function yToSvg(y) {
            return MARGIN.top + ((Y_MAX - y) / (Y_MAX - Y_MIN)) * plotHeight;
        }

        function svgToCue(event) {
            const point = svg.createSVGPoint();
            point.x = event.clientX;
            point.y = event.clientY;
            const transform = svg.getScreenCTM();
            if (!transform) return null;
            const local = point.matrixTransform(transform.inverse());
            const x = X_MIN + ((local.x - MARGIN.left) / plotWidth) * (X_MAX - X_MIN);
            const y = Y_MAX - ((local.y - MARGIN.top) / plotHeight) * (Y_MAX - Y_MIN);
            return [
                Math.max(X_MIN, Math.min(X_MAX, x)),
                Math.max(Y_MIN, Math.min(Y_MAX, y)),
            ];
        }

        function scoresAt(point) {
            return PATTERNS.map((pattern) => pattern[0] * point[0] + pattern[1] * point[1]);
        }

        function softmaxWeights(point) {
            const logits = scoresAt(point).map((score) => beta * score);
            const maxLogit = Math.max(...logits);
            const exponentials = logits.map((logit) => Math.exp(logit - maxLogit));
            const total = exponentials.reduce((sum, value) => sum + value, 0);
            return exponentials.map((value) => value / total);
        }

        function energy(point) {
            const scores = scoresAt(point);
            const maxScore = Math.max(...scores);
            const shiftedSum = scores.reduce(
                (sum, score) => sum + Math.exp(beta * (score - maxScore)),
                0
            );
            const logSumExp = beta * maxScore + Math.log(shiftedSum);
            return 0.5 * (point[0] ** 2 + point[1] ** 2) - logSumExp / beta;
        }

        function retrievalStep(point) {
            const weights = softmaxWeights(point);
            return PATTERNS.reduce(
                (next, pattern, index) => [
                    next[0] + weights[index] * pattern[0],
                    next[1] + weights[index] * pattern[1],
                ],
                [0, 0]
            );
        }

        function effectiveMemories(weights) {
            const entropy = -weights.reduce(
                (sum, weight) => sum + (weight > 0 ? weight * Math.log(weight) : 0),
                0
            );
            return Math.exp(entropy);
        }

        function attractorFrom(point) {
            let current = point.slice();
            for (let step = 0; step < 80; step += 1) {
                const next = retrievalStep(current);
                if (Math.hypot(next[0] - current[0], next[1] - current[1]) < 1e-7) {
                    return next;
                }
                current = next;
            }
            return current;
        }

        function updateReadouts() {
            betaValue.textContent = beta.toFixed(2);
            const attractor = attractorFrom(cue);
            const effective = effectiveMemories(softmaxWeights(attractor));
            let label = "single-pattern retrieval";
            if (effective >= 2.65) {
                label = "global average";
            } else if (effective >= 1.55) {
                label = "subset average";
            }
            regimeOutput.textContent = `${label} · effective memories ${effective.toFixed(2)}`;
        }

        function renderLandscape() {
            const columns = 56;
            const rows = 40;
            const samples = [];
            for (let row = 0; row < rows; row += 1) {
                for (let column = 0; column < columns; column += 1) {
                    const x = X_MIN + ((column + 0.5) / columns) * (X_MAX - X_MIN);
                    const y = Y_MAX - ((row + 0.5) / rows) * (Y_MAX - Y_MIN);
                    samples.push({ row, column, value: energy([x, y]) });
                }
            }

            const ordered = samples.map((sample) => sample.value).sort((a, b) => a - b);
            const low = ordered[0];
            const high = ordered[Math.floor(0.94 * (ordered.length - 1))];
            const span = Math.max(high - low, 1e-9);
            const bands = [
                COLORS.blue,
                COLORS.blueMid,
                COLORS.cyan,
                COLORS.teal,
                COLORS.grid,
                COLORS.snow2,
                COLORS.snow3,
            ];
            const cellWidth = plotWidth / columns;
            const cellHeight = plotHeight / rows;

            const cells = samples
                .map((sample) => {
                    const normalized = Math.max(0, Math.min(0.999999, (sample.value - low) / span));
                    const band = Math.min(bands.length - 1, Math.floor(normalized * bands.length));
                    const x = MARGIN.left + sample.column * cellWidth;
                    const y = MARGIN.top + sample.row * cellHeight;
                    return `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${(
                        cellWidth + 0.35
                    ).toFixed(2)}" height="${(cellHeight + 0.35).toFixed(2)}" fill="${
                        bands[band]
                    }"/>`;
                })
                .join("");

            svg.innerHTML = `
                <title>Modern Hopfield energy landscape</title>
                <desc>The exact log-sum-exp energy for three stored two-dimensional patterns. Click or drag to place the cue, then use the step or play controls to apply retrieval updates.</desc>
                <rect x="0" y="0" width="${WIDTH}" height="${HEIGHT}" fill="${COLORS.white}"/>
                <g shape-rendering="crispEdges">${cells}</g>
                <rect x="${MARGIN.left}" y="${MARGIN.top}" width="${plotWidth}" height="${plotHeight}"
                    fill="none" stroke="${COLORS.axis}" stroke-width="1"/>
                <line x1="${xToSvg(0)}" y1="${MARGIN.top}" x2="${xToSvg(0)}"
                    y2="${HEIGHT - MARGIN.bottom}" stroke="${COLORS.grid}" stroke-width="1"/>
                <line x1="${MARGIN.left}" y1="${yToSvg(0)}" x2="${WIDTH - MARGIN.right}"
                    y2="${yToSvg(0)}" stroke="${COLORS.grid}" stroke-width="1"/>
                <g data-hopfield-overlay></g>
            `;
            renderState();
        }

        function renderState() {
            const overlay = svg.querySelector("[data-hopfield-overlay]");
            if (!overlay) return;

            const pathPoints = trajectory
                .map((point) => `${xToSvg(point[0]).toFixed(2)},${yToSvg(point[1]).toFixed(2)}`)
                .join(" ");
            const patternMarks = PATTERNS.map(
                (pattern, index) => `
                    <circle cx="${xToSvg(pattern[0])}" cy="${yToSvg(pattern[1])}" r="7"
                        fill="${COLORS.ink}" stroke="${COLORS.white}" stroke-width="2"/>
                    <text x="${xToSvg(pattern[0]) + 11}" y="${yToSvg(pattern[1]) - 9}"
                        fill="${COLORS.ink}" font-size="12">x${index + 1}</text>
                `
            ).join("");
            overlay.innerHTML = `
                ${pathPoints ? `<polyline points="${pathPoints}" fill="none" stroke="${COLORS.red}"
                    stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>` : ""}
                ${trajectory
                    .slice(0, -1)
                    .map(
                        (point) =>
                            `<circle cx="${xToSvg(point[0])}" cy="${yToSvg(point[1])}" r="2.5"
                                fill="${COLORS.red}"/>`
                    )
                    .join("")}
                ${patternMarks}
                <circle cx="${xToSvg(cue[0])}" cy="${yToSvg(cue[1])}" r="7.5"
                    fill="${COLORS.yellow}" stroke="${COLORS.ink}" stroke-width="2"/>
                <text x="${xToSvg(cue[0]) + 11}" y="${yToSvg(cue[1]) + 4}"
                    fill="${COLORS.ink}" font-size="13">ξ</text>
            `;
            const description = svg.querySelector("desc");
            if (description) {
                description.textContent = `Modern Hopfield energy at beta ${beta.toFixed(
                    2
                )}, with cue coordinates ${cue[0].toFixed(2)}, ${cue[1].toFixed(2)}.`;
            }
            updateReadouts();
        }

        function stopPlay() {
            playToken += 1;
            playing = false;
            playButton.setAttribute("aria-pressed", "false");
            playButton.setAttribute("aria-label", "Play retrieval steps");
        }

        function advanceOnce() {
            const next = retrievalStep(cue);
            const distance = Math.hypot(next[0] - cue[0], next[1] - cue[1]);
            cue = next;
            trajectory.push(cue.slice());
            renderState();
            return distance;
        }

        function play() {
            if (playing) {
                stopPlay();
                return;
            }
            playing = true;
            playButton.setAttribute("aria-pressed", "true");
            playButton.setAttribute("aria-label", "Stop retrieval steps");
            const token = ++playToken;
            const maxSteps = 14;

            if (reducedMotion.matches) {
                for (let index = 0; index < maxSteps; index += 1) {
                    if (advanceOnce() < 1e-4) break;
                }
                stopPlay();
                return;
            }

            let completed = 0;
            const tick = () => {
                if (!playing || token !== playToken) return;
                const distance = advanceOnce();
                completed += 1;
                if (distance < 1e-4 || completed >= maxSteps) {
                    stopPlay();
                    return;
                }
                window.setTimeout(tick, 210);
            };
            tick();
        }

        betaInput.addEventListener("input", () => {
            stopPlay();
            beta = sanitizeBeta(betaInput.value);
            trajectory = [cue.slice()];
            renderLandscape();
        });

        playButton.addEventListener("click", play);
        stepButton.addEventListener("click", () => {
            stopPlay();
            advanceOnce();
        });
        resetButton.addEventListener("click", () => {
            stopPlay();
            cue = [...INITIAL_CUE];
            trajectory = [cue.slice()];
            renderState();
        });

        svg.addEventListener("pointerdown", (event) => {
            const nextCue = svgToCue(event);
            if (!nextCue) return;
            stopPlay();
            dragging = true;
            svg.setPointerCapture(event.pointerId);
            cue = nextCue;
            trajectory = [cue.slice()];
            renderState();
            event.preventDefault();
        });
        svg.addEventListener("pointermove", (event) => {
            if (!dragging) return;
            const nextCue = svgToCue(event);
            if (!nextCue) return;
            cue = nextCue;
            trajectory = [cue.slice()];
            renderState();
            event.preventDefault();
        });
        const endDrag = (event) => {
            dragging = false;
            if (svg.hasPointerCapture(event.pointerId)) {
                svg.releasePointerCapture(event.pointerId);
            }
        };
        svg.addEventListener("pointerup", endDrag);
        svg.addEventListener("pointercancel", endDrag);

        playButton.setAttribute("aria-pressed", "false");
        renderLandscape();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init, { once: true });
    } else {
        init();
    }
})();
