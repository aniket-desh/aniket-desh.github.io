(() => {
    "use strict";

    const SVG_NS = "http://www.w3.org/2000/svg";
    const WIDTH = 720;
    const HEIGHT = 430;
    const SCORES = [1.2, 0.35, -0.8];
    const GAP_SCALE = 4.4;
    const VERTICES = [
        { x: 360, y: 34 },
        { x: 78, y: 382 },
        { x: 642, y: 382 },
    ];
    const HEAT_COLORS = ["#eceff4", "#e5e9f0", "#d8dee9", "#88c0d0", "#81a1c1"];

    function init() {
        const root = document.getElementById("free-energy-root");
        const host = document.getElementById("free-energy-svg");
        const betaInput = document.getElementById("free-energy-beta");
        const betaOutput = document.getElementById("free-energy-beta-value");
        const gapOutput = document.getElementById("free-energy-gap");

        if (!root || !host || !betaInput || !betaOutput || !gapOutput) return;
        if (root.dataset.freeEnergyInitialized === "true") return;
        root.dataset.freeEnergyInitialized = "true";

        const svg = host instanceof SVGSVGElement
            ? host
            : document.createElementNS(SVG_NS, "svg");

        if (svg !== host) host.replaceChildren(svg);
        else svg.replaceChildren();

        svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
        svg.setAttribute("role", "img");
        svg.setAttribute("aria-labelledby", "free-energy-svg-title free-energy-svg-desc");
        svg.style.display = "block";
        svg.style.width = "100%";
        svg.style.maxWidth = "100%";
        svg.style.background = "#ffffff";
        svg.style.touchAction = "none";

        const title = append(svg, "title", { id: "free-energy-svg-title" });
        title.textContent = "Free-energy landscape on a three-token probability simplex";
        const description = append(svg, "desc", { id: "free-energy-svg-desc" });
        append(svg, "rect", {
            x: 0,
            y: 0,
            width: WIDTH,
            height: HEIGHT,
            fill: "#ffffff",
        });

        const fieldLayer = append(svg, "g", { "aria-hidden": "true" });
        const borderLayer = append(svg, "g", { "aria-hidden": "true" });
        const markerLayer = append(svg, "g");

        let candidate = [1 / 3, 1 / 3, 1 / 3];
        let optimum = [1 / 3, 1 / 3, 1 / 3];
        let beta = readPositive(betaInput.value, 1);
        let dragging = false;
        let candidateNode = null;
        let pendingFrame = 0;

        function renderField() {
            fieldLayer.replaceChildren();
            optimum = softmax(SCORES, beta);
            const resolution = 28;

            function latticePoint(i, j) {
                return [i / resolution, j / resolution, 1 - (i + j) / resolution];
            }

            for (let i = 0; i < resolution; i += 1) {
                for (let j = 0; j < resolution - i; j += 1) {
                    drawCell([
                        latticePoint(i, j),
                        latticePoint(i + 1, j),
                        latticePoint(i, j + 1),
                    ], GAP_SCALE);

                    if (i + j <= resolution - 2) {
                        drawCell([
                            latticePoint(i + 1, j),
                            latticePoint(i + 1, j + 1),
                            latticePoint(i, j + 1),
                        ], GAP_SCALE);
                    }
                }
            }

            borderLayer.replaceChildren();
            append(borderLayer, "polygon", {
                points: VERTICES.map((point) => `${point.x},${point.y}`).join(" "),
                fill: "none",
                stroke: "#2e3440",
                "stroke-width": 2,
                "stroke-linejoin": "round",
            });

            VERTICES.forEach((point, index) => {
                const offsets = [
                    { x: 0, y: -12 },
                    { x: -15, y: 20 },
                    { x: 15, y: 20 },
                ];
                const label = append(borderLayer, "text", {
                    x: point.x + offsets[index].x,
                    y: point.y + offsets[index].y,
                    fill: "#4c566a",
                    "font-family": "system-ui, sans-serif",
                    "font-size": 14,
                    "text-anchor": "middle",
                });
                label.textContent = String(index + 1);
            });

            updateMarkers();

            function drawCell(distributions, scale) {
                const centroid = [0, 1, 2].map(
                    (index) => distributions.reduce((sum, point) => sum + point[index], 0) / 3
                );
                const gap = freeEnergyGap(centroid, optimum, beta);
                const normalized = scale > 0 ? Math.sqrt(Math.min(1, gap / scale)) : 0;
                const colorIndex = Math.min(
                    HEAT_COLORS.length - 1,
                    Math.floor(normalized * HEAT_COLORS.length)
                );
                const fill = HEAT_COLORS[colorIndex];
                append(fieldLayer, "polygon", {
                    points: distributions
                        .map(barycentricToPoint)
                        .map((point) => `${point.x},${point.y}`)
                        .join(" "),
                    fill,
                    stroke: fill,
                    "stroke-width": 0.8,
                });
            }
        }

        function updateMarkers(restoreFocus = false) {
            const hadFocus = restoreFocus && document.activeElement === candidateNode;
            markerLayer.replaceChildren();

            const optimumPoint = barycentricToPoint(optimum);
            const optimumMarker = append(markerLayer, "path", {
                d: diamondPath(optimumPoint.x, optimumPoint.y, 9),
                fill: "#5e81ac",
                stroke: "#ffffff",
                "stroke-width": 3,
                "paint-order": "stroke",
                "aria-hidden": "true",
            });
            const optimumTitle = append(optimumMarker, "title");
            optimumTitle.textContent = `Softmax minimizer: ${formatDistribution(optimum)}`;

            const candidatePoint = barycentricToPoint(candidate);
            candidateNode = append(markerLayer, "g", {
                role: "slider",
                tabindex: 0,
                "aria-label": "Candidate probability distribution",
                "aria-valuetext": formatDistribution(candidate),
            });
            append(candidateNode, "circle", {
                cx: candidatePoint.x,
                cy: candidatePoint.y,
                r: 10,
                fill: "#ffffff",
                stroke: "#d08770",
                "stroke-width": 4,
            });
            append(candidateNode, "circle", {
                cx: candidatePoint.x,
                cy: candidatePoint.y,
                r: 2.5,
                fill: "#2e3440",
            });
            const candidateTitle = append(candidateNode, "title");
            candidateTitle.textContent = `Candidate distribution: ${formatDistribution(candidate)}`;
            candidateNode.addEventListener("keydown", onCandidateKeydown);

            const gap = freeEnergyGap(candidate, optimum, beta);
            betaOutput.textContent = beta.toFixed(2);
            gapOutput.textContent = gap.toFixed(3);
            description.textContent =
                `Three fixed token scores are ${SCORES.join(", ")}. ` +
                `At inverse temperature ${beta.toFixed(2)}, the softmax minimizer is ` +
                `${formatDistribution(optimum)}. The draggable candidate is ` +
                `${formatDistribution(candidate)}, with free-energy gap ${gap.toFixed(3)}.`;

            if (hadFocus) candidateNode.focus();
        }

        function onCandidateKeydown(event) {
            const point = barycentricToPoint(candidate);
            const step = event.shiftKey ? 3 : 10;
            let dx = 0;
            let dy = 0;

            if (event.key === "ArrowLeft") dx = -step;
            else if (event.key === "ArrowRight") dx = step;
            else if (event.key === "ArrowUp") dy = -step;
            else if (event.key === "ArrowDown") dy = step;
            else return;

            event.preventDefault();
            candidate = pointToDistribution(point.x + dx, point.y + dy);
            updateMarkers(true);
        }

        function updateCandidateFromPointer(event) {
            const point = clientToSvgPoint(svg, event.clientX, event.clientY);
            candidate = pointToDistribution(point.x, point.y);
            updateMarkers();
        }

        svg.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 && event.pointerType === "mouse") return;
            dragging = true;
            svg.setPointerCapture?.(event.pointerId);
            updateCandidateFromPointer(event);
        });
        svg.addEventListener("pointermove", (event) => {
            if (dragging) updateCandidateFromPointer(event);
        });
        svg.addEventListener("pointerup", (event) => {
            dragging = false;
            svg.releasePointerCapture?.(event.pointerId);
        });
        svg.addEventListener("pointercancel", () => {
            dragging = false;
        });

        betaInput.addEventListener("input", () => {
            beta = readPositive(betaInput.value, beta);
            if (pendingFrame) cancelAnimationFrame(pendingFrame);
            pendingFrame = requestAnimationFrame(() => {
                pendingFrame = 0;
                renderField();
            });
        });

        const resize = () => {
            const availableWidth = root.clientWidth || host.clientWidth || WIDTH;
            svg.style.height = `${Math.round(availableWidth * HEIGHT / WIDTH)}px`;
        };
        resize();
        if ("ResizeObserver" in window) {
            const observer = new ResizeObserver(resize);
            observer.observe(root);
        } else {
            window.addEventListener("resize", resize, { passive: true });
        }

        renderField();
    }

    function append(parent, name, attributes = {}) {
        const element = document.createElementNS(SVG_NS, name);
        Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
        parent.appendChild(element);
        return element;
    }

    function softmax(scores, beta) {
        const scaled = scores.map((score) => beta * score);
        const maximum = Math.max(...scaled);
        const exponentials = scaled.map((score) => Math.exp(score - maximum));
        const total = exponentials.reduce((sum, value) => sum + value, 0);
        return exponentials.map((value) => value / total);
    }

    function freeEnergyGap(distribution, optimum, beta) {
        let divergence = 0;
        for (let index = 0; index < distribution.length; index += 1) {
            const probability = distribution[index];
            if (probability > 0) divergence += probability * Math.log(probability / optimum[index]);
        }
        return Math.max(0, divergence / beta);
    }

    function barycentricToPoint(distribution) {
        return {
            x: distribution.reduce((sum, probability, index) => sum + probability * VERTICES[index].x, 0),
            y: distribution.reduce((sum, probability, index) => sum + probability * VERTICES[index].y, 0),
        };
    }

    function pointToDistribution(x, y) {
        const [a, b, c] = VERTICES;
        const denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        const p0 = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / denominator;
        const p1 = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / denominator;
        return projectOntoSimplex([p0, p1, 1 - p0 - p1]);
    }

    function projectOntoSimplex(values) {
        const sorted = values.slice().sort((left, right) => right - left);
        let cumulative = 0;
        let threshold = 0;

        for (let index = 0; index < sorted.length; index += 1) {
            cumulative += sorted[index];
            const candidate = (cumulative - 1) / (index + 1);
            if (sorted[index] - candidate > 0) threshold = candidate;
        }

        const projected = values.map((value) => Math.max(value - threshold, 0));
        const total = projected.reduce((sum, value) => sum + value, 0);
        return total > 0
            ? projected.map((value) => value / total)
            : [1 / 3, 1 / 3, 1 / 3];
    }

    function clientToSvgPoint(svg, clientX, clientY) {
        const matrix = svg.getScreenCTM();
        if (matrix) {
            const point = new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse());
            return { x: point.x, y: point.y };
        }
        const bounds = svg.getBoundingClientRect();
        return {
            x: (clientX - bounds.left) * WIDTH / Math.max(bounds.width, 1),
            y: (clientY - bounds.top) * HEIGHT / Math.max(bounds.height, 1),
        };
    }

    function diamondPath(x, y, radius) {
        return `M ${x} ${y - radius} L ${x + radius} ${y} L ${x} ${y + radius} L ${x - radius} ${y} Z`;
    }

    function formatDistribution(distribution) {
        return distribution.map((value) => value.toFixed(2)).join(", ");
    }

    function readPositive(value, fallback) {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init, { once: true });
    } else {
        init();
    }
})();
