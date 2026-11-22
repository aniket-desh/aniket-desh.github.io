document.addEventListener('DOMContentLoaded', function() {
    var NS = 'http://www.w3.org/2000/svg';
    var nord = {
        ink: '#2e3440',
        muted: '#4c566a',
        line: '#d8dee9',
        pale: '#eceff4',
        blue: '#5e81ac',
        cyan: '#88c0d0',
        green: '#a3be8c',
        red: '#bf616a',
        orange: '#d08770',
        purple: '#b48ead',
        white: '#fff'
    };

    function svgElement(name, attrs, textValue) {
        var el = document.createElementNS(NS, name);
        Object.keys(attrs || {}).forEach(function(key) {
            el.setAttribute(key, attrs[key]);
        });
        if (textValue !== undefined) el.textContent = textValue;
        return el;
    }

    // Markov's two-state model for vowel-consonant transitions in Eugene Onegin.
    (function() {
        var svg = document.getElementById('markov-svg');
        if (!svg) return;

        var playButton = document.getElementById('markov-play');
        var status = document.getElementById('markov-status');
        var detail = document.getElementById('markov-detail');
        var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        var transitions = {
            V: { V: 0.128, C: 0.872 },
            C: { V: 0.663, C: 0.337 }
        };
        var sequence = ['C'];
        var seed = 1;
        var running = false;
        var transitionPaths = {};

        function random() {
            seed = (1664525 * seed + 1013904223) >>> 0;
            return seed / 4294967296;
        }

        function sampleNext(current) {
            return random() < transitions[current].V ? 'V' : 'C';
        }

        function className(letter) {
            return letter === 'V' ? 'vowel' : 'consonant';
        }

        function addMarker(defs, id, color) {
            var marker = svgElement('marker', {
                id: id,
                viewBox: '0 0 10 10',
                refX: 9,
                refY: 5,
                markerWidth: 6,
                markerHeight: 6,
                orient: 'auto-start-reverse'
            });
            marker.appendChild(svgElement('path', {
                d: 'M 0 0 L 10 5 L 0 10 z',
                fill: color
            }));
            defs.appendChild(marker);
        }

        function drawTransition(key, d, label, labelX, labelY, activeKey) {
            var active = key === activeKey;
            var path = svgElement('path', {
                d: d,
                fill: 'none',
                stroke: active ? nord.red : nord.line,
                'stroke-width': active ? 3 : 2,
                'marker-end': active ? 'url(#markov-arrow-active)' : 'url(#markov-arrow)'
            });
            svg.appendChild(path);
            transitionPaths[key] = path;
            svg.appendChild(svgElement('text', {
                x: labelX,
                y: labelY,
                'text-anchor': 'middle',
                'font-size': 12,
                fill: nord.muted
            }, label));
        }

        function draw(activeKey) {
            while (svg.lastChild) svg.removeChild(svg.lastChild);
            transitionPaths = {};

            var width = Math.max(300, Math.round(svg.clientWidth));
            var height = Math.round(svg.clientHeight);
            var xV = Math.max(80, Math.round(width * 0.3));
            var xC = Math.min(width - 80, Math.round(width * 0.7));
            var stateY = 105;
            var radius = 35;
            svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);

            var title = svgElement('title', { id: 'markov-title' }, "Markov's vowel-consonant transition model");
            var desc = svgElement('desc', { id: 'markov-desc' }, 'From a vowel, the next letter is a consonant with probability 0.872. From a consonant, the next letter is a vowel with probability 0.663. The row below shows one sampled sequence.');
            svg.appendChild(title);
            svg.appendChild(desc);

            var defs = svgElement('defs');
            addMarker(defs, 'markov-arrow', nord.line);
            addMarker(defs, 'markov-arrow-active', nord.red);
            svg.appendChild(defs);

            drawTransition(
                'V-V',
                'M ' + (xV - 26) + ' ' + (stateY - 24) + ' C ' + (xV - 68) + ' ' + (stateY - 58) + ', ' + (xV - 68) + ' ' + (stateY + 58) + ', ' + (xV - 26) + ' ' + (stateY + 24),
                '0.128', Math.max(24, xV - 72), stateY + 4, activeKey
            );
            drawTransition(
                'C-C',
                'M ' + (xC + 26) + ' ' + (stateY + 24) + ' C ' + (xC + 68) + ' ' + (stateY + 58) + ', ' + (xC + 68) + ' ' + (stateY - 58) + ', ' + (xC + 26) + ' ' + (stateY - 24),
                '0.337', Math.min(width - 24, xC + 72), stateY + 4, activeKey
            );
            drawTransition(
                'V-C',
                'M ' + (xV + radius) + ' ' + (stateY - 12) + ' Q ' + ((xV + xC) / 2) + ' 54, ' + (xC - radius) + ' ' + (stateY - 12),
                '0.872', (xV + xC) / 2, 67, activeKey
            );
            drawTransition(
                'C-V',
                'M ' + (xC - radius) + ' ' + (stateY + 12) + ' Q ' + ((xV + xC) / 2) + ' 156, ' + (xV + radius) + ' ' + (stateY + 12),
                '0.663', (xV + xC) / 2, 151, activeKey
            );

            [
                { letter: 'V', label: 'vowel', x: xV, color: nord.blue },
                { letter: 'C', label: 'consonant', x: xC, color: nord.orange }
            ].forEach(function(state) {
                var current = sequence[sequence.length - 1] === state.letter;
                svg.appendChild(svgElement('circle', {
                    cx: state.x,
                    cy: stateY,
                    r: radius,
                    fill: state.color,
                    stroke: current ? nord.ink : state.color,
                    'stroke-width': current ? 3 : 2
                }));
                svg.appendChild(svgElement('text', {
                    x: state.x,
                    y: stateY + 4,
                    'text-anchor': 'middle',
                    'font-size': 13,
                    'font-weight': 600,
                    fill: nord.white
                }, state.label));
            });

            var sequenceY = height - 25;
            var tokenGap = 31;
            var maxVisible = Math.max(6, Math.floor((width - 42) / tokenGap));
            var visible = sequence.slice(-maxVisible);
            var rowWidth = visible.length * tokenGap;
            var startX = (width - rowWidth) / 2 + tokenGap / 2;
            svg.appendChild(svgElement('text', {
                x: 16,
                y: sequenceY - 28,
                'font-size': 12,
                fill: nord.muted
            }, sequence.length > visible.length ? 'one sampled sequence (latest states)' : 'one sampled sequence'));
            visible.forEach(function(letter, index) {
                var latest = index === visible.length - 1;
                svg.appendChild(svgElement('circle', {
                    cx: startX + index * tokenGap,
                    cy: sequenceY,
                    r: 11,
                    fill: letter === 'V' ? nord.blue : nord.orange,
                    stroke: latest ? nord.ink : nord.white,
                    'stroke-width': latest ? 2 : 1
                }));
                svg.appendChild(svgElement('text', {
                    x: startX + index * tokenGap,
                    y: sequenceY + 4,
                    'text-anchor': 'middle',
                    'font-size': 11,
                    'font-weight': 600,
                    fill: nord.white
                }, letter));
            });
        }

        function updateText(from, to) {
            var probability = transitions[from][to];
            status.textContent = 'Current class: ' + className(to);
            detail.textContent = 'From a ' + className(from) + ', this step sampled a ' + className(to) + ' (' + (probability * 100).toFixed(1) + '%).';
        }

        function animateTransition(from, to, done) {
            var key = from + '-' + to;
            draw(key);
            var path = transitionPaths[key];
            if (reducedMotion.matches || !path) {
                done();
                return;
            }

            var dot = svgElement('circle', {
                r: 5,
                fill: nord.red,
                stroke: nord.white,
                'stroke-width': 2
            });
            svg.appendChild(dot);
            var length = path.getTotalLength();
            var startTime = null;

            function frame(timestamp) {
                if (startTime === null) startTime = timestamp;
                var progress = Math.min(1, (timestamp - startTime) / 420);
                var point = path.getPointAtLength(length * progress);
                dot.setAttribute('cx', point.x);
                dot.setAttribute('cy', point.y);
                if (progress < 1) {
                    requestAnimationFrame(frame);
                } else {
                    done();
                }
            }
            requestAnimationFrame(frame);
        }

        function finish() {
            running = false;
            playButton.disabled = false;
            playButton.textContent = 'Replay sequence';
            draw();
        }

        function step() {
            if (sequence.length >= 12) {
                finish();
                return;
            }
            var from = sequence[sequence.length - 1];
            var to = sampleNext(from);
            animateTransition(from, to, function() {
                sequence.push(to);
                updateText(from, to);
                draw();
                window.setTimeout(step, reducedMotion.matches ? 0 : 90);
            });
        }

        playButton.addEventListener('click', function() {
            if (running) return;
            sequence = ['C'];
            seed = 1;
            running = true;
            playButton.disabled = true;
            playButton.textContent = 'Building sequence...';
            status.textContent = 'Current class: consonant';
            detail.textContent = 'Sampling each next class from the probabilities above.';
            draw();
            step();
        });

        var resizeFrame = null;
        window.addEventListener('resize', function() {
            if (running) return;
            if (resizeFrame) cancelAnimationFrame(resizeFrame);
            resizeFrame = requestAnimationFrame(function() { draw(); });
        }, { passive: true });
        draw();
    })();

    // Exact congestion accounting on the matching space of a 2 x 3 ladder.
    (function() {
        var svg = document.getElementById('congestion-svg');
        if (!svg) return;

        var deterministicButton = document.getElementById('congestion-deterministic');
        var fractionalButton = document.getElementById('congestion-fractional');
        var status = document.getElementById('congestion-status');
        var detail = document.getElementById('congestion-detail');
        var mode = 'deterministic';

        var graphEdges = [
            { u: 0, v: 1, label: 'a' },
            { u: 1, v: 2, label: 'b' },
            { u: 3, v: 4, label: 'c' },
            { u: 4, v: 5, label: 'd' },
            { u: 0, v: 3, label: 'e' },
            { u: 1, v: 4, label: 'f' },
            { u: 2, v: 5, label: 'g' }
        ];

        function bitCount(mask) {
            var count = 0;
            while (mask) {
                count += mask & 1;
                mask >>>= 1;
            }
            return count;
        }

        function isMatching(mask) {
            var used = {};
            for (var i = 0; i < graphEdges.length; i++) {
                if (!(mask & (1 << i))) continue;
                var edge = graphEdges[i];
                if (used[edge.u] || used[edge.v]) return false;
                used[edge.u] = true;
                used[edge.v] = true;
            }
            return true;
        }

        function matchingLabel(mask) {
            if (mask === 0) return '\u2205';
            var label = '';
            for (var i = 0; i < graphEdges.length; i++) {
                if (mask & (1 << i)) label += graphEdges[i].label;
            }
            return label;
        }

        var states = [];
        for (var mask = 0; mask < (1 << graphEdges.length); mask++) {
            if (isMatching(mask)) states.push(mask);
        }
        var stateIndex = new Map();
        states.forEach(function(state, index) { stateIndex.set(state, index); });

        function transitionKey(a, b) {
            var ia = stateIndex.get(a);
            var ib = stateIndex.get(b);
            return ia < ib ? ia + ':' + ib : ib + ':' + ia;
        }

        function addLoad(loads, key, amount) {
            loads.set(key, (loads.get(key) || 0) + amount);
        }

        var deterministicLoads = new Map();
        var fractionalLoads = new Map();

        states.forEach(function(start) {
            states.forEach(function(target) {
                if (start === target) return;

                var remove = [];
                var add = [];
                for (var i = 0; i < graphEdges.length; i++) {
                    var inStart = Boolean(start & (1 << i));
                    var inTarget = Boolean(target & (1 << i));
                    if (inStart && !inTarget) remove.push(i);
                    if (!inStart && inTarget) add.push(i);
                }

                var current = start;
                remove.concat(add).forEach(function(edgeIndex) {
                    var next = current ^ (1 << edgeIndex);
                    addLoad(deterministicLoads, transitionKey(current, next), 1);
                    current = next;
                });

                var paths = [];
                function enumerateValidOrders(state, remaining, path) {
                    if (remaining.length === 0) {
                        paths.push(path.slice());
                        return;
                    }
                    remaining.forEach(function(edgeIndex, position) {
                        var next = state ^ (1 << edgeIndex);
                        if (!isMatching(next)) return;
                        var rest = remaining.slice(0, position).concat(remaining.slice(position + 1));
                        path.push(transitionKey(state, next));
                        enumerateValidOrders(next, rest, path);
                        path.pop();
                    });
                }
                enumerateValidOrders(start, remove.concat(add), []);
                var weight = 1 / paths.length;
                paths.forEach(function(path) {
                    path.forEach(function(key) { addLoad(fractionalLoads, key, weight); });
                });
            });
        });

        var transitions = [];
        for (var i = 0; i < states.length; i++) {
            for (var j = i + 1; j < states.length; j++) {
                var diff = states[i] ^ states[j];
                if (diff && (diff & (diff - 1)) === 0) {
                    transitions.push({ a: i, b: j, key: i + ':' + j });
                }
            }
        }

        function appendTitleAndDescription() {
            svg.appendChild(svgElement('title', { id: 'congestion-title' }, 'Congestion in a small matching state space'));
            svg.appendChild(svgElement('desc', { id: 'congestion-desc' }, 'Nodes are matchings of a two by three ladder. Edge width is the exact all-pairs load divided by transition capacity.'));
        }

        function drawLadderKey(width) {
            var keyWidth = Math.min(170, width - 80);
            var left = (width - keyWidth) / 2;
            var dx = keyWidth / 2;
            var top = 25;
            var bottom = 57;
            var points = [
                [left, top], [left + dx, top], [left + 2 * dx, top],
                [left, bottom], [left + dx, bottom], [left + 2 * dx, bottom]
            ];

            svg.appendChild(svgElement('text', {
                x: width / 2,
                y: 12,
                'text-anchor': 'middle',
                'font-size': 11,
                fill: nord.muted
            }, 'edge labels on G'));

            graphEdges.forEach(function(edge) {
                var p1 = points[edge.u];
                var p2 = points[edge.v];
                svg.appendChild(svgElement('line', {
                    x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
                    stroke: nord.line,
                    'stroke-width': 2
                }));
                var mx = (p1[0] + p2[0]) / 2;
                var my = (p1[1] + p2[1]) / 2;
                var vertical = Math.abs(p1[0] - p2[0]) < 1;
                svg.appendChild(svgElement('text', {
                    x: mx + (vertical ? 9 : 0),
                    y: my + (vertical ? 3 : -5),
                    'text-anchor': 'middle',
                    'font-size': 10,
                    fill: nord.muted
                }, edge.label));
            });
            points.forEach(function(point) {
                svg.appendChild(svgElement('circle', {
                    cx: point[0], cy: point[1], r: 3,
                    fill: nord.ink
                }));
            });
        }

        function draw() {
            while (svg.firstChild) svg.removeChild(svg.firstChild);
            appendTitleAndDescription();

            var width = Math.max(300, Math.round(svg.clientWidth));
            var height = Math.round(svg.clientHeight);
            svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
            drawLadderKey(width);

            var groups = [[], [], [], []];
            states.forEach(function(state, index) {
                groups[bitCount(state)].push(index);
            });
            groups.forEach(function(group) {
                group.sort(function(a, b) {
                    return matchingLabel(states[a]).localeCompare(matchingLabel(states[b]));
                });
            });

            var positions = [];
            var graphTop = 105;
            var graphBottom = height - 18;
            var graphHeight = graphBottom - graphTop;
            groups.forEach(function(group, cardinality) {
                var x = 28 + cardinality * (width - 56) / 3;
                svg.appendChild(svgElement('text', {
                    x: x,
                    y: 91,
                    'text-anchor': 'middle',
                    'font-size': 10,
                    fill: nord.muted
                }, '|M|=' + cardinality));
                group.forEach(function(stateIndexValue, row) {
                    positions[stateIndexValue] = {
                        x: x,
                        y: graphTop + (row + 1) * graphHeight / (group.length + 1)
                    };
                });
            });

            var loads = mode === 'deterministic' ? deterministicLoads : fractionalLoads;
            var maxLoad = 0;
            loads.forEach(function(value) { maxLoad = Math.max(maxLoad, value); });
            var maxKeys = [];
            loads.forEach(function(value, key) {
                if (Math.abs(value - maxLoad) < 1e-8) maxKeys.push(key);
            });

            transitions.forEach(function(transition) {
                var p1 = positions[transition.a];
                var p2 = positions[transition.b];
                var load = loads.get(transition.key) || 0;
                var ratio = load / maxLoad;
                var isMaximum = maxKeys.indexOf(transition.key) !== -1;
                svg.appendChild(svgElement('line', {
                    x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y,
                    stroke: isMaximum ? nord.red : (mode === 'deterministic' ? nord.blue : nord.green),
                    'stroke-width': (0.7 + 4.7 * ratio).toFixed(2),
                    'stroke-opacity': isMaximum ? 0.95 : (0.22 + 0.58 * ratio).toFixed(2),
                    'stroke-linecap': 'round'
                }));
            });

            states.forEach(function(state, index) {
                var point = positions[index];
                svg.appendChild(svgElement('circle', {
                    cx: point.x, cy: point.y, r: 10.5,
                    fill: nord.white,
                    stroke: nord.muted,
                    'stroke-width': 1.1
                }));
                svg.appendChild(svgElement('text', {
                    x: point.x,
                    y: point.y + 3,
                    'text-anchor': 'middle',
                    'font-size': matchingLabel(state).length > 2 ? 7.5 : 9,
                    fill: nord.ink
                }, matchingLabel(state)));
            });

            var congestion = maxLoad * (2 * graphEdges.length / states.length);
            var endpoints = maxKeys[0].split(':').map(Number);
            status.textContent = 'max congestion \u03c1 = ' + congestion.toFixed(1);
            detail.textContent = states.length + ' states \u00b7 ' + transitions.length + ' transitions \u00b7 bottleneck ' +
                matchingLabel(states[endpoints[0]]) + ' \u2194 ' + matchingLabel(states[endpoints[1]]);
            svg.setAttribute('aria-label', status.textContent + '. ' + detail.textContent + '.');
        }

        function setMode(nextMode) {
            mode = nextMode;
            var deterministic = mode === 'deterministic';
            deterministicButton.classList.toggle('is-active', deterministic);
            fractionalButton.classList.toggle('is-active', !deterministic);
            deterministicButton.setAttribute('aria-pressed', String(deterministic));
            fractionalButton.setAttribute('aria-pressed', String(!deterministic));
            draw();
        }

        deterministicButton.addEventListener('click', function() { setMode('deterministic'); });
        fractionalButton.addEventListener('click', function() { setMode('fractional'); });

        var resizeFrame = null;
        window.addEventListener('resize', function() {
            if (resizeFrame) cancelAnimationFrame(resizeFrame);
            resizeFrame = requestAnimationFrame(draw);
        }, { passive: true });
        draw();
    })();

    // Step-through of the local transport and additive lifting argument.
    (function() {
        var svg = document.getElementById('transport-svg');
        if (!svg) return;

        var previousButton = document.getElementById('transport-prev');
        var nextButton = document.getElementById('transport-next');
        var status = document.getElementById('transport-status');
        var detail = document.getElementById('transport-detail');
        var step = 0;
        var details = [
            'Independent matchings can differ on several alternating components.',
            'A global canonical path processes the full symmetric difference.',
            'Conditioning compares the distributions with one edge absent or present.',
            'The coupling makes the sampled pair differ only on the component B containing e.',
            'Local inequalities accumulate additively into a global spectral bound.'
        ];

        var ladderEdges = [
            { id: 't01', u: 0, v: 1 }, { id: 't12', u: 1, v: 2 },
            { id: 't23', u: 2, v: 3 }, { id: 't34', u: 3, v: 4 },
            { id: 'b01', u: 5, v: 6 }, { id: 'b12', u: 6, v: 7 },
            { id: 'b23', u: 7, v: 8 }, { id: 'b34', u: 8, v: 9 },
            { id: 'r0', u: 0, v: 5 }, { id: 'r1', u: 1, v: 6 },
            { id: 'r2', u: 2, v: 7 }, { id: 'r3', u: 3, v: 8 },
            { id: 'r4', u: 4, v: 9 }
        ];
        var edgeById = {};
        ladderEdges.forEach(function(edge) { edgeById[edge.id] = edge; });
        var matchingX = ['r0', 'r2', 'r4'];
        var matchingY = ['t01', 'b01', 't34', 'b34'];
        var globalOrder = ['r0', 'r2', 'r4', 't01', 'b01', 't34', 'b34'];

        function clearSvg() {
            while (svg.firstChild) svg.removeChild(svg.firstChild);
            svg.appendChild(svgElement('title', { id: 'transport-title' }, 'From global paths to local transport'));
            svg.appendChild(svgElement('desc', { id: 'transport-desc' }, details[step]));
        }

        function graphGeometry(width, height) {
            var margin = Math.max(34, width * 0.09);
            var dx = (width - 2 * margin) / 4;
            var top = Math.max(76, height * 0.29);
            var bottom = Math.min(height - 62, height * 0.67);
            var points = [];
            for (var column = 0; column < 5; column++) {
                points.push([margin + column * dx, top]);
            }
            for (var lowerColumn = 0; lowerColumn < 5; lowerColumn++) {
                points.push([margin + lowerColumn * dx, bottom]);
            }
            return { points: points, margin: margin, dx: dx, top: top, bottom: bottom };
        }

        function drawBase(points, opacity) {
            ladderEdges.forEach(function(edge) {
                var p1 = points[edge.u];
                var p2 = points[edge.v];
                svg.appendChild(svgElement('line', {
                    x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
                    stroke: nord.line,
                    'stroke-width': 2,
                    'stroke-opacity': opacity
                }));
            });
            points.forEach(function(point) {
                svg.appendChild(svgElement('circle', {
                    cx: point[0], cy: point[1], r: 4,
                    fill: nord.ink,
                    'fill-opacity': opacity
                }));
            });
        }

        function drawGraphEdge(edgeId, points, color, width, opacity, dash) {
            var edge = edgeById[edgeId];
            var p1 = points[edge.u];
            var p2 = points[edge.v];
            var attrs = {
                x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
                stroke: color,
                'stroke-width': width,
                'stroke-opacity': opacity,
                'stroke-linecap': 'round'
            };
            if (dash) attrs['stroke-dasharray'] = dash;
            svg.appendChild(svgElement('line', attrs));
        }

        function midpoint(edgeId, points) {
            var edge = edgeById[edgeId];
            var p1 = points[edge.u];
            var p2 = points[edge.v];
            return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
        }

        function drawLegend(width) {
            var start = width / 2 - 78;
            svg.appendChild(svgElement('line', {
                x1: start, y1: 22, x2: start + 22, y2: 22,
                stroke: nord.blue, 'stroke-width': 5, 'stroke-linecap': 'round'
            }));
            svg.appendChild(svgElement('text', {
                x: start + 29, y: 26, 'font-size': 11, fill: nord.muted
            }, 'X'));
            svg.appendChild(svgElement('line', {
                x1: start + 80, y1: 22, x2: start + 102, y2: 22,
                stroke: nord.orange, 'stroke-width': 5, 'stroke-linecap': 'round'
            }));
            svg.appendChild(svgElement('text', {
                x: start + 109, y: 26, 'font-size': 11, fill: nord.muted
            }, 'Y'));
        }

        function drawOrderBadge(edgeId, order, points) {
            var mid = midpoint(edgeId, points);
            var edge = edgeById[edgeId];
            var vertical = Math.abs(points[edge.u][0] - points[edge.v][0]) < 1;
            var x = mid[0] + (vertical ? 13 : 0);
            var y = mid[1] + (vertical ? 0 : -12);
            svg.appendChild(svgElement('circle', {
                cx: x, cy: y, r: 8,
                fill: nord.white,
                stroke: nord.muted,
                'stroke-width': 1
            }));
            svg.appendChild(svgElement('text', {
                x: x, y: y + 3,
                'text-anchor': 'middle',
                'font-size': 9,
                fill: nord.ink
            }, String(order)));
        }

        function drawGlobalPair(width, height, showOrder) {
            var geometry = graphGeometry(width, height);
            drawBase(geometry.points, 1);
            drawLegend(width);

            matchingX.forEach(function(edgeId) {
                drawGraphEdge(edgeId, geometry.points, nord.blue, 6, 0.95);
            });
            matchingY.forEach(function(edgeId) {
                drawGraphEdge(edgeId, geometry.points, nord.orange, 6, 0.95);
            });

            if (!showOrder) {
                var boxes = [
                    { x: geometry.margin - 18, width: geometry.dx + 36, label: 'B\u2081' },
                    { x: geometry.margin + 2 * geometry.dx - 20, width: 40, label: 'B\u2082' },
                    { x: geometry.margin + 3 * geometry.dx - 18, width: geometry.dx + 36, label: 'B\u2083' }
                ];
                boxes.forEach(function(box) {
                    svg.appendChild(svgElement('rect', {
                        x: box.x,
                        y: geometry.top - 24,
                        width: box.width,
                        height: geometry.bottom - geometry.top + 48,
                        rx: 10,
                        fill: 'none',
                        stroke: nord.purple,
                        'stroke-width': 1.5,
                        'stroke-dasharray': '5 4'
                    }));
                    svg.appendChild(svgElement('text', {
                        x: box.x + box.width / 2,
                        y: geometry.bottom + 43,
                        'text-anchor': 'middle',
                        'font-size': 11,
                        fill: nord.muted
                    }, box.label));
                });
            } else {
                globalOrder.forEach(function(edgeId, index) {
                    drawOrderBadge(edgeId, index + 1, geometry.points);
                });
                svg.appendChild(svgElement('text', {
                    x: width / 2,
                    y: height - 16,
                    'text-anchor': 'middle',
                    'font-size': 12,
                    fill: nord.muted
                }, '|X \u25b3 Y| = 7 transitions'));
            }
        }

        function drawCondition(width, height) {
            var geometry = graphGeometry(width, height);
            drawBase(geometry.points, 0.7);
            drawGraphEdge('r0', geometry.points, nord.red, 8, 1);
            var mid = midpoint('r0', geometry.points);
            svg.appendChild(svgElement('circle', {
                cx: mid[0], cy: mid[1], r: 18,
                fill: 'none',
                stroke: nord.red,
                'stroke-width': 1.5,
                'stroke-dasharray': '4 3'
            }));
            svg.appendChild(svgElement('text', {
                x: width / 2,
                y: 34,
                'text-anchor': 'middle',
                'font-size': 13,
                fill: nord.ink
            }, '\u03bc\u03c4(e=0)  \u2194  \u03bc\u03c4(e=1)'));
            svg.appendChild(svgElement('text', {
                x: mid[0] - 18,
                y: mid[1] + 4,
                'text-anchor': 'end',
                'font-size': 12,
                fill: nord.red
            }, 'e'));
        }

        function drawLocalTransport(width, height) {
            var geometry = graphGeometry(width, height);
            drawBase(geometry.points, 0.55);
            drawGraphEdge('r0', geometry.points, nord.blue, 6, 1);
            drawGraphEdge('t01', geometry.points, nord.orange, 6, 1);
            drawGraphEdge('b01', geometry.points, nord.orange, 6, 1);
            drawOrderBadge('r0', 1, geometry.points);
            drawOrderBadge('t01', 2, geometry.points);
            drawOrderBadge('b01', 3, geometry.points);
            svg.appendChild(svgElement('rect', {
                x: geometry.margin - 18,
                y: geometry.top - 28,
                width: geometry.dx + 36,
                height: geometry.bottom - geometry.top + 56,
                rx: 10,
                fill: 'none',
                stroke: nord.green,
                'stroke-width': 2,
                'stroke-dasharray': '5 4'
            }));
            svg.appendChild(svgElement('text', {
                x: geometry.margin + geometry.dx / 2,
                y: geometry.bottom + 48,
                'text-anchor': 'middle',
                'font-size': 12,
                fill: nord.muted
            }, 'B = X \u25b3 Y contains e'));
        }

        function drawArrow(x1, y1, x2, y2) {
            svg.appendChild(svgElement('line', {
                x1: x1, y1: y1, x2: x2, y2: y2,
                stroke: nord.muted,
                'stroke-width': 1.5
            }));
            var angle = Math.atan2(y2 - y1, x2 - x1);
            var size = 6;
            var points = [
                [x2, y2],
                [x2 - size * Math.cos(angle - Math.PI / 6), y2 - size * Math.sin(angle - Math.PI / 6)],
                [x2 - size * Math.cos(angle + Math.PI / 6), y2 - size * Math.sin(angle + Math.PI / 6)]
            ].map(function(point) { return point[0] + ',' + point[1]; }).join(' ');
            svg.appendChild(svgElement('polygon', { points: points, fill: nord.muted }));
        }

        function drawLift(width, height) {
            var localY = 58;
            var sumY = height * 0.55;
            var globalY = height - 36;
            var count = width < 420 ? 4 : 5;
            var margin = width < 420 ? 34 : 50;
            for (var i = 0; i < count; i++) {
                var x = margin + i * (width - 2 * margin) / (count - 1);
                svg.appendChild(svgElement('circle', {
                    cx: x, cy: localY, r: 24,
                    fill: nord.white,
                    stroke: nord.blue,
                    'stroke-width': 2
                }));
                svg.appendChild(svgElement('text', {
                    x: x, y: localY - 2,
                    'text-anchor': 'middle',
                    'font-size': 10,
                    fill: nord.ink
                }, 'local'));
                svg.appendChild(svgElement('text', {
                    x: x, y: localY + 11,
                    'text-anchor': 'middle',
                    'font-size': 10,
                    fill: nord.muted
                }, 'edge e' + (i + 1)));
                drawArrow(x, localY + 26, width / 2, sumY - 25);
            }
            svg.appendChild(svgElement('ellipse', {
                cx: width / 2, cy: sumY, rx: 76, ry: 24,
                fill: nord.pale,
                stroke: nord.green,
                'stroke-width': 2
            }));
            svg.appendChild(svgElement('text', {
                x: width / 2, y: sumY + 4,
                'text-anchor': 'middle',
                'font-size': 12,
                fill: nord.ink
            }, 'add local decay'));
            drawArrow(width / 2, sumY + 26, width / 2, globalY - 22);
            svg.appendChild(svgElement('ellipse', {
                cx: width / 2, cy: globalY, rx: 64, ry: 21,
                fill: nord.blue,
                stroke: nord.blue,
                'stroke-width': 2
            }));
            svg.appendChild(svgElement('text', {
                x: width / 2, y: globalY + 4,
                'text-anchor': 'middle',
                'font-size': 12,
                fill: nord.pale
            }, 'global gap'));
        }

        function draw() {
            clearSvg();
            var width = Math.max(300, Math.round(svg.clientWidth));
            var height = Math.round(svg.clientHeight);
            svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);

            if (step === 0) drawGlobalPair(width, height, false);
            if (step === 1) drawGlobalPair(width, height, true);
            if (step === 2) drawCondition(width, height);
            if (step === 3) drawLocalTransport(width, height);
            if (step === 4) drawLift(width, height);

            status.textContent = (step + 1) + ' / ' + details.length;
            detail.textContent = details[step];
            previousButton.disabled = step === 0;
            nextButton.disabled = step === details.length - 1;
            svg.setAttribute('aria-label', details[step]);
        }

        previousButton.addEventListener('click', function() {
            if (step > 0) {
                step--;
                draw();
            }
        });
        nextButton.addEventListener('click', function() {
            if (step < details.length - 1) {
                step++;
                draw();
            }
        });

        var resizeFrame = null;
        window.addEventListener('resize', function() {
            if (resizeFrame) cancelAnimationFrame(resizeFrame);
            resizeFrame = requestAnimationFrame(draw);
        }, { passive: true });
        draw();
    })();
});
