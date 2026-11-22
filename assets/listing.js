// Tag filter for the post listing pages: /blog/ and /drosophila/.
document.querySelectorAll('.tag-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.tag-btn').forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');
        var tag = btn.getAttribute('data-tag');
        document.querySelectorAll('.post-entry').forEach(function(entry) {
            if (tag === 'all' || entry.getAttribute('data-tag') === tag) {
                entry.classList.remove('hidden');
            } else {
                entry.classList.add('hidden');
            }
        });
    });
});
