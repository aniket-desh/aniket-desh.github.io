(() => {
    "use strict";

    const COLORS = Object.freeze({
        white: "#ffffff",
        ink: "#2e3440",
        ink2: "#3b4252",
        axis: "#4c566a",
        grid: "#d8dee9",
        snow2: "#e5e9f0",
        snow3: "#eceff4",
        blue: "#5e81ac",
        blueMid: "#81a1c1",
        cyan: "#88c0d0",
        teal: "#8fbcbb",
        orange: "#d08770",
        red: "#bf616a",
        purple: "#b48ead",
    });

    const N = 64;
    const D = 11;
    const FIT_MIN = 0;
    const FIT_MAX = 10;
    const SVG_NS = "http://www.w3.org/2000/svg";
    const LOG_FACTORIAL = (() => {
        const values = new Array(N + 1).fill(0);
        for (let value = 1; value <= N; value += 1) {
            values[value] = values[value - 1] + Math.log(value);
        }
        return values;
    })();

    function logBinomial(n, k) {
        if (k < 0 || k > n) return -Infinity;
        return LOG_FACTORIAL[n] - LOG_FACTORIAL[k] - LOG_FACTORIAL[n - k];
    }

    function logAddExp(left, right) {
        if (left === -Infinity) return right;
        if (right === -Infinity) return left;
        const high = Math.max(left, right);
        const low = Math.min(left, right);
        return high + Math.log1p(Math.exp(low - high));
    }

    function admissible(t, a, b) {
        return a + b <= D && t - a + b <= D;
    }

    function logOverlap(t) {
        if (t < 0 || t > N || t > 2 * D) return -Infinity;
        let total = -Infinity;
        for (let a = 0; a <= t; a += 1) {
            for (let b = 0; b <= N - t; b += 1) {
                if (!admissible(t, a, b)) continue;
                const term = logBinomial(t, a) + logBinomial(N - t, b);
                total = logAddExp(total, term);
            }
        }
        return total;
    }

    function linearFit(points) {
        const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
        const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
        const numerator = points.reduce(
            (sum, point) => sum + (point.x - meanX) * (point.y - meanY),
            0
        );
        const denominator = points.reduce(
            (sum, point) => sum + (point.x - meanX) ** 2,
            0
        );
        const slope = numerator / denominator;
        return { slope, intercept: meanY - slope * meanX };
    }

    function init() {
        const root = document.getElementById("sdm-combinatorics-root");
        if (!root) return;

        const distanceInput = root.querySelector("#sdm-distance");
        const distanceValue = root.querySelector("#sdm-distance-value");
        const overlapValue = root.querySelector("#sdm-overlap-value");
        const latticeSvg = root.querySelector("#sdm-lattice-svg");
        const overlapSvg = root.querySelector("#sdm-overlap-svg");
        if (!distanceInput || !distanceValue || !overlapValue || !latticeSvg || !overlapSvg) {
            return;
        }

        const exactPoints = [];
        for (let t = 0; t <= 2 * D; t += 1) {
            exactPoints.push({ x: t, y: logOverlap(t) });
        }
        const fit = linearFit(
            exactPoints.filter((point) => point.x >= FIT_MIN && point.x <= FIT_MAX)
        );
        const fitAt = (t) => fit.intercept + fit.slope * t;

        function sanitizeDistance(value) {
            const parsed = Math.round(Number(value));
            if (!Number.isFinite(parsed)) return 0;
            return Math.max(0, Math.min(N, parsed));
        }

        function configureSvg(svg, label, width, height) {
            svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
            svg.setAttribute("role", "img");
            svg.setAttribute("aria-label", label);
            svg.style.width = "100%";
            svg.style.height = "auto";
            svg.style.display = "block";
            svg.style.background = COLORS.white;
        }

        configureSvg(
            latticeSvg,
            "Admissible combinatorial terms in the intersection of two Hamming balls",
            520,
            350
        );
        configureSvg(
            overlapSvg,
            "Exact logarithmic Hamming-ball overlap, close-distance linear fit, and fit residual",
            720,
            440
        );

        function renderLattice(t) {
            const width = 520;
            const height = 350;
            const margin = { top: 30, right: 20, bottom: 48, left: 52 };
            const plotWidth = width - margin.left - margin.right;
            const plotHeight = height - margin.top - margin.bottom;
            const xMax = t <= 2 * D ? Math.max(1, Math.min(t, D)) : D;
            const yMax = D;
            const xScale = (value) => margin.left + (value / xMax) * plotWidth;
            const yScale = (value) =>
                margin.top + ((yMax - value) / Math.max(1, yMax)) * plotHeight;
            const terms = [];
            let maxTerm = -Infinity;

            if (t <= 2 * D) {
                for (let a = 0; a <= Math.min(t, D); a += 1) {
                    for (let b = 0; b <= D; b += 1) {
                        if (!admissible(t, a, b)) continue;
                        const logTerm = logBinomial(t, a) + logBinomial(N - t, b);
                        terms.push({ a, b, logTerm });
                        maxTerm = Math.max(maxTerm, logTerm);
                    }
                }
            }

            const gridPoints = [];
            for (let a = 0; a <= xMax; a += 1) {
                for (let b = 0; b <= yMax; b += 1) {
                    gridPoints.push(
                        `<circle cx="${xScale(a)}" cy="${yScale(b)}" r="1.5" fill="${COLORS.grid}"/>`
                    );
                }
            }
            const termMarks = terms
                .map((term) => {
                    const relative = Math.exp(0.5 * (term.logTerm - maxTerm));
                    const radius = 3 + 6 * relative;
                    return `<circle cx="${xScale(term.a)}" cy="${yScale(term.b)}"
                        r="${radius.toFixed(2)}" fill="${COLORS.blue}"
                        stroke="${COLORS.white}" stroke-width="1"/>`;
                })
                .join("");

            const xTickStep = xMax <= 6 ? 1 : 2;
            const xTicks = [];
            for (let value = 0; value <= xMax; value += xTickStep) {
                xTicks.push(
                    `<line x1="${xScale(value)}" y1="${height - margin.bottom}" x2="${xScale(
                        value
                    )}" y2="${height - margin.bottom + 5}" stroke="${COLORS.axis}"/>`,
                    `<text x="${xScale(value)}" y="${height - margin.bottom + 19}"
                        text-anchor="middle" fill="${COLORS.axis}" font-size="11">${value}</text>`
                );
            }
            const yTicks = [0, 5, 10, 11]
                .map(
                    (value) =>
                        `<line x1="${margin.left - 5}" y1="${yScale(value)}" x2="${margin.left}"
                            y2="${yScale(value)}" stroke="${COLORS.axis}"/>` +
                        `<text x="${margin.left - 9}" y="${yScale(value) + 4}" text-anchor="end"
                            fill="${COLORS.axis}" font-size="11">${value}</text>`
                )
                .join("");

            const emptyMessage =
                t > 2 * D
                    ? `<text x="${margin.left + plotWidth / 2}" y="${
                          margin.top + plotHeight / 2
                      }" text-anchor="middle" fill="${COLORS.red}">
                          no overlap
                      </text>`
                    : "";

            latticeSvg.innerHTML = `
                <title>Admissible overlap terms at Hamming distance ${t}</title>
                <desc>Each blue point is an admissible pair a, b satisfying both Hamming-ball radius constraints. Its area reflects its relative combinatorial contribution.</desc>
                <rect x="0" y="0" width="${width}" height="${height}" fill="${COLORS.white}"/>
                <rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}"
                    fill="${COLORS.white}" stroke="${COLORS.axis}" stroke-width="1"/>
                ${gridPoints.join("")}
                ${termMarks}
                ${emptyMessage}
                ${xTicks.join("")}
                ${yTicks}
                <text x="${margin.left + plotWidth / 2}" y="${height - 8}" text-anchor="middle"
                    fill="${COLORS.axis}">a</text>
                <text x="14" y="${margin.top + plotHeight / 2}" text-anchor="middle"
                    transform="rotate(-90 14 ${margin.top + plotHeight / 2})"
                    fill="${COLORS.axis}">b</text>
            `;
        }

        function renderOverlap(t) {
            const width = 720;
            const height = 440;
            const left = 56;
            const right = 24;
            const plotWidth = width - left - right;
            const topPlot = { top: 30, bottom: 264 };
            const residualPlot = { top: 320, bottom: 405 };
            const xScale = (value) => left + (value / (2 * D)) * plotWidth;
            const logMin = 12.5;
            const logMax = 28.5;
            const yLog = (value) =>
                topPlot.bottom -
                ((value - logMin) / (logMax - logMin)) * (topPlot.bottom - topPlot.top);
            const residuals = exactPoints.map((point) => ({
                x: point.x,
                y: point.y - fitAt(point.x),
            }));
            const residualMin = Math.min(...residuals.map((point) => point.y), -0.5) - 0.2;
            const residualMax = Math.max(...residuals.map((point) => point.y), 0.4) + 0.2;
            const yResidual = (value) =>
                residualPlot.bottom -
                ((value - residualMin) / (residualMax - residualMin)) *
                    (residualPlot.bottom - residualPlot.top);
            const path = (points, xAccessor, yAccessor) =>
                points
                    .map(
                        (point, index) =>
                            `${index === 0 ? "M" : "L"} ${xAccessor(point.x).toFixed(
                                2
                            )} ${yAccessor(point.y).toFixed(2)}`
                    )
                    .join(" ");
            const exactPath = path(exactPoints, xScale, yLog);
            const fitPoints = [
                { x: 0, y: fitAt(0) },
                { x: 2 * D, y: fitAt(2 * D) },
            ];
            const fitPath = path(fitPoints, xScale, yLog);
            const residualPath = path(residuals, xScale, yResidual);
            const fitShadeX = xScale(FIT_MIN);
            const fitShadeWidth = xScale(FIT_MAX) - fitShadeX;
            const xTicks = [0, 5, 10, 15, 20, 22]
                .map(
                    (value) =>
                        `<line x1="${xScale(value)}" y1="${residualPlot.bottom}" x2="${xScale(
                            value
                        )}" y2="${residualPlot.bottom + 5}" stroke="${COLORS.axis}"/>` +
                        `<text x="${xScale(value)}" y="${residualPlot.bottom + 19}"
                            text-anchor="middle" fill="${COLORS.axis}" font-size="11">${value}</text>`
                )
                .join("");
            const logTicks = [14, 18, 22, 26]
                .map(
                    (value) =>
                        `<line x1="${left - 5}" y1="${yLog(value)}" x2="${left}"
                            y2="${yLog(value)}" stroke="${COLORS.axis}"/>` +
                        `<line x1="${left}" y1="${yLog(value)}" x2="${width - right}"
                            y2="${yLog(value)}" stroke="${COLORS.snow2}"/>` +
                        `<text x="${left - 9}" y="${yLog(value) + 4}" text-anchor="end"
                            fill="${COLORS.axis}" font-size="11">${value}</text>`
                )
                .join("");
            const residualTicks = [-4, -2, 0]
                .filter((value) => value >= residualMin && value <= residualMax)
                .map(
                    (value) =>
                        `<line x1="${left - 5}" y1="${yResidual(value)}" x2="${left}"
                            y2="${yResidual(value)}" stroke="${COLORS.axis}"/>` +
                        `<line x1="${left}" y1="${yResidual(value)}" x2="${width - right}"
                            y2="${yResidual(value)}" stroke="${
                                value === 0 ? COLORS.axis : COLORS.snow2
                            }"/>` +
                        `<text x="${left - 9}" y="${yResidual(value) + 4}" text-anchor="end"
                            fill="${COLORS.axis}" font-size="11">${value}</text>`
                )
                .join("");

            let selectedMarks = "";
            if (t <= 2 * D) {
                const selectedLog = exactPoints[t].y;
                const selectedResidual = selectedLog - fitAt(t);
                selectedMarks = `
                    <line x1="${xScale(t)}" y1="${topPlot.top}" x2="${xScale(t)}"
                        y2="${residualPlot.bottom}" stroke="${COLORS.red}" stroke-width="1"/>
                    <circle cx="${xScale(t)}" cy="${yLog(selectedLog)}" r="5"
                        fill="${COLORS.red}" stroke="${COLORS.white}" stroke-width="1.5"/>
                    <circle cx="${xScale(t)}" cy="${yResidual(selectedResidual)}" r="4"
                        fill="${COLORS.red}" stroke="${COLORS.white}" stroke-width="1.2"/>
                `;
            } else {
                selectedMarks = `
                    <text x="${width - right}" y="17" text-anchor="end"
                        fill="${COLORS.red}">I = 0</text>
                `;
            }

            overlapSvg.innerHTML = `
                <title>Exact Hamming-ball overlap and close-distance approximation</title>
                <desc>The upper plot gives the exact log overlap for n equals ${N} and d equals ${D}. A linear model is fit only from t equals ${FIT_MIN} through ${FIT_MAX}. The lower plot shows exact log overlap minus that fit. Overlap is exactly zero beyond t equals ${
                2 * D
            }.</desc>
                <rect x="0" y="0" width="${width}" height="${height}" fill="${COLORS.white}"/>
                <rect x="${fitShadeX}" y="${topPlot.top}" width="${fitShadeWidth}"
                    height="${topPlot.bottom - topPlot.top}" fill="${COLORS.snow3}"/>
                <rect x="${fitShadeX}" y="${residualPlot.top}" width="${fitShadeWidth}"
                    height="${residualPlot.bottom - residualPlot.top}" fill="${COLORS.snow3}"/>
                ${logTicks}
                ${residualTicks}
                <line x1="${left}" y1="${topPlot.bottom}" x2="${width - right}"
                    y2="${topPlot.bottom}" stroke="${COLORS.axis}"/>
                <line x1="${left}" y1="${topPlot.top}" x2="${left}" y2="${topPlot.bottom}"
                    stroke="${COLORS.axis}"/>
                <line x1="${left}" y1="${residualPlot.bottom}" x2="${width - right}"
                    y2="${residualPlot.bottom}" stroke="${COLORS.axis}"/>
                <line x1="${left}" y1="${residualPlot.top}" x2="${left}"
                    y2="${residualPlot.bottom}" stroke="${COLORS.axis}"/>
                <path d="${exactPath}" fill="none" stroke="${COLORS.blue}" stroke-width="2.4"
                    stroke-linecap="round" stroke-linejoin="round"/>
                ${exactPoints
                    .map(
                        (point) =>
                            `<circle cx="${xScale(point.x)}" cy="${yLog(point.y)}" r="2.7"
                                fill="${COLORS.blue}"/>`
                    )
                    .join("")}
                <path d="${fitPath}" fill="none" stroke="${COLORS.orange}" stroke-width="2"
                    stroke-dasharray="6 5"/>
                <path d="${residualPath}" fill="none" stroke="${COLORS.purple}" stroke-width="2"
                    stroke-linecap="round" stroke-linejoin="round"/>
                ${residuals
                    .map(
                        (point) =>
                            `<circle cx="${xScale(point.x)}" cy="${yResidual(point.y)}" r="2.4"
                                fill="${COLORS.purple}"/>`
                    )
                    .join("")}
                ${selectedMarks}
                ${xTicks}
                <text x="13" y="${(topPlot.top + topPlot.bottom) / 2}" text-anchor="middle"
                    transform="rotate(-90 13 ${(topPlot.top + topPlot.bottom) / 2})"
                    fill="${COLORS.axis}">log I</text>
                <text x="13" y="${(residualPlot.top + residualPlot.bottom) / 2}" text-anchor="middle"
                    transform="rotate(-90 13 ${(residualPlot.top + residualPlot.bottom) / 2})"
                    fill="${COLORS.axis}">resid.</text>
                <text x="${left + plotWidth / 2}" y="${height - 7}" text-anchor="middle"
                    fill="${COLORS.axis}">t</text>
            `;
        }

        function render() {
            const t = sanitizeDistance(distanceInput.value);
            distanceValue.textContent = String(t);
            const overlap = logOverlap(t);
            overlapValue.textContent =
                overlap === -Infinity ? "−∞ (I = 0)" : overlap.toFixed(3);
            renderLattice(t);
            renderOverlap(t);
        }

        distanceInput.addEventListener("input", render);
        render();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init, { once: true });
    } else {
        init();
    }
})();
