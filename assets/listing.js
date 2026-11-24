// Tag filter for the post listing pages: /blog/ and /drosophila/.
document.querySelectorAll('.tag-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.tag-btn').forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var tag = btn.getAttribute('data-tag');
        document.querySelectorAll('.post-entry').forEach(function(entry) {
            // data-tag may list several space-separated tags.
            var tags = (entry.getAttribute('data-tag') || '').split(/\s+/);
            if (tag === 'all' || tags.indexOf(tag) !== -1) {
                entry.classList.remove('hidden');
            } else {
                entry.classList.add('hidden');
            }
        });
    });
});
