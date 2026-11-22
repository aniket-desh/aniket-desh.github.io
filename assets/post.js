// Shared behaviour for blog posts: footnotes and table-of-contents highlighting.
// Load it at the end of <body>, after any script of the post's own.
(function() {
    'use strict';

    // Turn a footnote marker into a numbered link with a hover tooltip, and give
    // the matching list item its number and a link back to the marker.
    function linkMarker(sup, n) {
        sup.id = 'fnref:' + n;

        var link = document.createElement('a');
        link.href = '#fn:' + n;
        link.className = 'footnote-ref';
        link.textContent = '[' + n + ']';
        sup.appendChild(link);
    }

    function linkItem(sup, li, n) {
        var tooltip = document.createElement('div');
        tooltip.className = 'footnote-tooltip';
        tooltip.innerHTML = '<sup>' + n + '</sup> ' + li.innerHTML;
        sup.appendChild(tooltip);

        li.id = 'fn:' + n;
        var numSup = document.createElement('sup');
        numSup.textContent = n;
        var backLink = document.createElement('a');
        backLink.href = '#fnref:' + n;
        backLink.className = 'footnote-backref';
        backLink.textContent = '↩';

        var original = li.innerHTML;
        li.innerHTML = '';
        li.appendChild(numSup);
        li.insertAdjacentHTML('beforeend', ' ' + original + ' ');
        li.appendChild(backLink);
    }

    // Keyed footnotes: <sup data-fn="key"></sup> in the text and <li data-fn="key">
    // in the list. Numbers follow first appearance, and the list is reordered to match.
    function keyedFootnotes(footnotes) {
        var container = document.querySelector('.container');
        if (!container) return;

        var items = {};
        footnotes.querySelectorAll('li[data-fn]').forEach(function(li) {
            items[li.getAttribute('data-fn')] = li;
        });

        var counter = 0;
        var assigned = {};
        container.querySelectorAll('sup[data-fn]').forEach(function(sup) {
            var key = sup.getAttribute('data-fn');
            if (!key) return;

            if (!assigned[key]) {
                counter++;
                assigned[key] = counter;
            }
            var n = assigned[key];

            linkMarker(sup, n);
            var li = items[key];
            if (li && !li.id) linkItem(sup, li, n);
        });

        var ol = footnotes.querySelector('ol');
        Array.from(ol.querySelectorAll('li[id]')).sort(function(a, b) {
            return parseInt(a.id.replace('fn:', '')) - parseInt(b.id.replace('fn:', ''));
        }).forEach(function(li) { ol.appendChild(li); });
    }

    // Numbered footnotes: <sup>3</sup> in the text refers to the third list item.
    function numberedFootnotes(footnotes) {
        var body = document.querySelector('.post-body');
        if (!body) return;

        var items = footnotes.querySelectorAll('li');
        body.querySelectorAll('sup').forEach(function(sup) {
            var n = sup.textContent.trim();
            if (!/^\d+$/.test(n)) return;

            sup.textContent = '';
            linkMarker(sup, n);
            var li = items[parseInt(n) - 1];
            if (li) linkItem(sup, li, n);
        });
    }

    document.addEventListener('DOMContentLoaded', function() {
        var footnotes = document.querySelector('.footnotes');
        if (!footnotes) return;

        if (footnotes.querySelector('li[data-fn]')) keyedFootnotes(footnotes);
        else numberedFootnotes(footnotes);
    });

    // Table of contents: mark the link of the section being read. A section counts
    // as reached once its top is within `offset` pixels of the top of the viewport.
    // Two attributes on .toc-sidebar adjust this per post:
    //   data-toc-offset="200"      the offset in pixels (default 120)
    //   data-toc-default="first"   keep the first link marked above the first section
    var toc = document.querySelector('.toc-sidebar');
    if (!toc) return;

    var links = Array.from(toc.querySelectorAll('a'));
    var sections = links.map(function(a) {
        return document.querySelector(a.getAttribute('href'));
    });
    var offset = Number(toc.dataset.tocOffset || 120);
    var fallback = toc.dataset.tocDefault === 'first' ? 0 : null;

    function updateToc() {
        var active = fallback;
        sections.forEach(function(section, i) {
            if (section && section.getBoundingClientRect().top <= offset) active = i;
        });
        links.forEach(function(link, i) {
            link.classList.toggle('active', i === active);
        });
    }

    window.addEventListener('scroll', updateToc, { passive: true });
    updateToc();
})();
