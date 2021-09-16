// var disqus_shortname;
var disqus_identifier;

var disqus_config = function () {
    var disqus_thread = $("#disqus_thread");
    this.page.url = window.location.href;
    this.page.identifier = disqus_thread.data('disqusIdentifier');
};

(function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = 'https://mxnet.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
})();
