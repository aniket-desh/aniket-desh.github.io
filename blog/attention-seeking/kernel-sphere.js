(() => {
    "use strict";

    const SVG_NS = "http://www.w3.org/2000/svg";
    const WIDTH = 720;
    const HEIGHT = 430;
    const CENTER = { x: 360, y: 215 };
    const UNIT_RADIUS = 126;
    const VALUE_SCALE = 54;
    const KEY_ANGLES = [
        0.08, 0.52, 1.03, 1.47, 2.02, 2.54,
        3.08, 3.61, 4.07, 4.58, 5.13, 5.72,
    ];
    const VALUES = KEY_ANGLES.map(
        (angle) => 0.82 * Math.sin(2 * angle) + 0.32 * Math.cos(5 * angle)
    );

    function init() {
        const root = document.getElementById("kernel-sphere-root");
        const host = document.getElementById("kernel-sphere-svg");
        const tauInput = document.getElementById("kernel-tau");
        const tauOutput = document.getElementById("kernel-tau-value");
        const predictionOutput = document.getElementById("kernel-prediction");

        if (!root || !host || !tauInput || !tauOutput || !predictionOutput) return;
        if (root.dataset.kernelSphereInitialized === "true") return;
        root.dataset.kernelSphereInitialized = "true";

        const svg = host instanceof SVGSVGElement
            ? host
            : document.createElementNS(SVG_NS, "svg");

        if (svg !== host) host.replaceChildren(svg);
        else svg.replaceChildren();

        svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
        svg.setAttribute("role", "img");
        svg.setAttribute("aria-labelledby", "kernel-sphere-svg-title kernel-sphere-svg-desc");
        svg.style.display = "block";
        svg.style.width = "100%";
        svg.style.maxWidth = "100%";
        svg.style.background = "#ffffff";
        svg.style.touchAction = "none";

        const title = append(svg, "title", { id: "kernel-sphere-svg-title" });
        title.textContent = "Gaussian kernel regression on normalized keys";
        const description = append(svg, "desc", { id: "kernel-sphere-svg-desc" });
        append(svg, "rect", {
            x: 0,
            y: 0,
            width: WIDTH,
            height: HEIGHT,
            fill: "#ffffff",
        });

        const baseLayer = append(svg, "g", { "aria-hidden": "true" });
        const curveLayer = append(svg, "g", { "aria-hidden": "true" });
        const observationLayer = append(svg, "g", { "aria-hidden": "true" });
        const queryLayer = append(svg, "g");

        let tau = readPositive(tauInput.value, 0.35);
        let queryAngle = 0.82;
        let dragging = false;
        let queryNode = null;
        let pendingFrame = 0;

        append(baseLayer, "circle", {
            cx: CENTER.x,
            cy: CENTER.y,
            r: UNIT_RADIUS,
            fill: "none",
            stroke: "#d8dee9",
            "stroke-width": 2,
        });
        append(baseLayer, "circle", {
            cx: CENTER.x,
            cy: CENTER.y,
            r: 2.5,
            fill: "#4c566a",
        });

        function render() {
            const activeElementWasQuery = document.activeElement === queryNode;
            curveLayer.replaceChildren();
            observationLayer.replaceChildren();
            queryLayer.replaceChildren();

            const samples = 220;
            let path = "";
            for (let index = 0; index <= samples; index += 1) {
                const angle = 2 * Math.PI * index / samples;
                const prediction = kernelEstimate(angle, tau).prediction;
                const point = polarPoint(angle, UNIT_RADIUS + VALUE_SCALE * prediction);
                path += `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)} `;
            }
            append(curveLayer, "path", {
                d: `${path}Z`,
                fill: "none",
                stroke: "#5e81ac",
                "stroke-width": 3,
                "stroke-linejoin": "round",
                "stroke-linecap": "round",
            });

            const current = kernelEstimate(queryAngle, tau);
            const maximumWeight = Math.max(...current.weights);

            KEY_ANGLES.forEach((angle, index) => {
                const basePoint = polarPoint(angle, UNIT_RADIUS);
                const observedPoint = polarPoint(angle, UNIT_RADIUS + VALUE_SCALE * VALUES[index]);
                const relativeWeight = current.weights[index] / Math.max(maximumWeight, Number.MIN_VALUE);

                append(observationLayer, "line", {
                    x1: basePoint.x,
                    y1: basePoint.y,
                    x2: observedPoint.x,
                    y2: observedPoint.y,
                    stroke: "#4c566a",
                    "stroke-width": 1.5,
                    opacity: 0.45,
                });
                append(observationLayer, "circle", {
                    cx: basePoint.x,
                    cy: basePoint.y,
                    r: 2.5,
                    fill: "#2e3440",
                });
                append(observationLayer, "circle", {
                    cx: observedPoint.x,
                    cy: observedPoint.y,
                    r: 4.5 + 6.5 * relativeWeight,
                    fill: "#d08770",
                    stroke: "#ffffff",
                    "stroke-width": 2.5,
                    opacity: 0.55 + 0.45 * relativeWeight,
                });
            });

            const queryPoint = polarPoint(queryAngle, UNIT_RADIUS);
            const predictionPoint = polarPoint(
                queryAngle,
                UNIT_RADIUS + VALUE_SCALE * current.prediction
            );
            append(queryLayer, "line", {
                x1: CENTER.x,
                y1: CENTER.y,
                x2: predictionPoint.x,
                y2: predictionPoint.y,
                stroke: "#bf616a",
                "stroke-width": 2,
                opacity: 0.8,
                "aria-hidden": "true",
            });
            append(queryLayer, "circle", {
                cx: predictionPoint.x,
                cy: predictionPoint.y,
                r: 7,
                fill: "#5e81ac",
                stroke: "#ffffff",
                "stroke-width": 3,
                "paint-order": "stroke",
                "aria-hidden": "true",
            });

            queryNode = append(queryLayer, "g", {
                role: "slider",
                tabindex: 0,
                "aria-label": "Query angle",
                "aria-valuemin": 0,
                "aria-valuemax": 360,
                "aria-valuenow": radiansToDegrees(queryAngle).toFixed(0),
                "aria-valuetext": `${radiansToDegrees(queryAngle).toFixed(0)} degrees`,
            });
            append(queryNode, "path", {
                d: diamondPath(queryPoint.x, queryPoint.y, 10),
                fill: "#bf616a",
                stroke: "#ffffff",
                "stroke-width": 3,
                "paint-order": "stroke",
            });
            const queryTitle = append(queryNode, "title");
            queryTitle.textContent = `Query at ${radiansToDegrees(queryAngle).toFixed(0)} degrees`;
            queryNode.addEventListener("keydown", onQueryKeydown);

            tauOutput.textContent = tau.toFixed(2);
            predictionOutput.textContent = current.prediction.toFixed(3);
            description.textContent =
                `Twelve normalized keys lie around the unit circle. Their scalar observations are ` +
                `shown by radial stems, and marker size indicates the current normalized kernel weight. ` +
                `At bandwidth ${tau.toFixed(2)}, the query is at ` +
                `${radiansToDegrees(queryAngle).toFixed(0)} degrees and the weighted prediction is ` +
                `${current.prediction.toFixed(3)}.`;

            if (activeElementWasQuery) queryNode.focus();
        }

        function onQueryKeydown(event) {
            let direction = 0;
            if (event.key === "ArrowLeft" || event.key === "ArrowDown") direction = -1;
            else if (event.key === "ArrowRight" || event.key === "ArrowUp") direction = 1;
            else return;

            event.preventDefault();
            const step = event.shiftKey ? Math.PI / 180 : 5 * Math.PI / 180;
            queryAngle = normalizeAngle(queryAngle + direction * step);
            render();
        }

        function updateQueryFromPointer(event) {
            const point = clientToSvgPoint(svg, event.clientX, event.clientY);
            const dx = point.x - CENTER.x;
            const dy = point.y - CENTER.y;
            if (Math.hypot(dx, dy) < 8) return;
            queryAngle = normalizeAngle(Math.atan2(dy, dx));
            render();
        }

        svg.addEventListener("pointerdown", (event) => {
            if (event.button !== 0 && event.pointerType === "mouse") return;
            dragging = true;
            svg.setPointerCapture?.(event.pointerId);
            updateQueryFromPointer(event);
        });
        svg.addEventListener("pointermove", (event) => {
            if (dragging) updateQueryFromPointer(event);
        });
        svg.addEventListener("pointerup", (event) => {
            dragging = false;
            svg.releasePointerCapture?.(event.pointerId);
        });
        svg.addEventListener("pointercancel", () => {
            dragging = false;
        });

        tauInput.addEventListener("input", () => {
            tau = readPositive(tauInput.value, tau);
            if (pendingFrame) cancelAnimationFrame(pendingFrame);
            pendingFrame = requestAnimationFrame(() => {
                pendingFrame = 0;
                render();
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

        render();
    }

    function append(parent, name, attributes = {}) {
        const element = document.createElementNS(SVG_NS, name);
        Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
        parent.appendChild(element);
        return element;
    }

    function kernelEstimate(queryAngle, tau) {
        const logits = KEY_ANGLES.map(
            (keyAngle) => Math.cos(queryAngle - keyAngle) / tau
        );
        const maximum = Math.max(...logits);
        const exponentials = logits.map((logit) => Math.exp(logit - maximum));
        const total = exponentials.reduce((sum, value) => sum + value, 0);
        const weights = exponentials.map((value) => value / total);
        const prediction = weights.reduce(
            (sum, weight, index) => sum + weight * VALUES[index],
            0
        );
        return { weights, prediction };
    }

    function polarPoint(angle, radius) {
        return {
            x: CENTER.x + radius * Math.cos(angle),
            y: CENTER.y + radius * Math.sin(angle),
        };
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

    function normalizeAngle(angle) {
        const fullTurn = 2 * Math.PI;
        return ((angle % fullTurn) + fullTurn) % fullTurn;
    }

    function radiansToDegrees(angle) {
        return normalizeAngle(angle) * 180 / Math.PI;
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
