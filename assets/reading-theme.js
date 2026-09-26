(function () {
    'use strict';

    var storageKey = 'aniket-reading-theme';
    var root = document.documentElement;
    var preference = null;

    function validTheme(value) {
        return value === 'light' || value === 'dark';
    }

    try {
        var saved = localStorage.getItem(storageKey);
        if (validTheme(saved)) preference = saved;
    } catch (error) {
        // The toggle still works when browser storage is unavailable.
    }

    function applyTheme(theme) {
        root.dataset.theme = theme;
        root.style.colorScheme = theme;
        var action = 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode';
        document.querySelectorAll('.theme-toggle').forEach(function (button) {
            button.setAttribute('aria-label', action);
        });
    }

    function preferredTheme() {
        return preference || 'light';
    }

    // This script runs in the head so the chosen palette is ready before paint.
    applyTheme(preferredTheme());

    function bindControls() {
        document.querySelectorAll('.theme-toggle').forEach(function (button) {
            button.hidden = false;
            button.addEventListener('click', function () {
                preference = root.dataset.theme === 'dark' ? 'light' : 'dark';
                applyTheme(preference);
                try {
                    localStorage.setItem(storageKey, preference);
                } catch (error) {
                    // Keep the current page usable in private or restricted contexts.
                }
            });
        });
        applyTheme(preferredTheme());
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bindControls, { once: true });
    } else {
        bindControls();
    }

    window.addEventListener('storage', function (event) {
        if (event.key !== storageKey && event.key !== null) return;
        preference = validTheme(event.newValue) ? event.newValue : null;
        applyTheme(preferredTheme());
    });
})();
