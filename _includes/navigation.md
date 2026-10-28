{% for link in site.data.navigation.main %}
  <a class="normal" href="{{ link.url | relative_url }}"{% if page.layout == 'blog' and link.title == 'Blog' %} aria-current="page"{% endif %}>{{ link.title }}</a>
{% endfor %}

