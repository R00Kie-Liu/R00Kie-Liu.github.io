<h2 id="awards" style="margin: 60px 0px 10px;">Awards</h2>

<ul class="awards-list">
  <li><strong>[06/2025]</strong> Outstanding Ph.D. Graduate of SJTU.</li>
  <li><strong>[12/2023]</strong> Funded by China Scholarship Council (CSC) as a visiting researcher for 1 year</li>
  <li><strong>[09/2022]</strong> CETC scholarship for Ph.D. students in SJTU (¥20000).</li>
  <li><strong>[10/2019]</strong> Selected as member of Wu Wenjun (<a href="https://dzb.sjtu.edu.cn/Data/View/3875">吴文俊</a>) AI honorary PhD student in SJTU (< 1%).</li>
  <li class="awards-extra" style="display:none;"><strong>[06/2019]</strong> Outstanding undergraduate student of Xidian University (XDU)</li>
  <li class="awards-extra" style="display:none;"><strong>[2015-2018]</strong> First Prize Scholarship of Xidian University (< 4%), 3 times </li>
  <li class="awards-extra" style="display:none;"><strong>[2015-2018]</strong> Outstanding Student of Xidian University (< 3%), 2 times</li>
  <li class="awards-extra" style="display:none;"><strong>[10/2017]</strong> First prize of National college students Mathematical Contest in Shaanxi (< 3%)</li>
</ul>

<div class="awards-toggle" style="text-align:center; margin-top: 8px;">
  <button onclick="toggleAwards()" id="awardsToggleBtn" style="
    background: none;
    border: 1px solid var(--border-color);
    color: var(--link-color);
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    font-family: inherit;
  " onmouseover="this.style.background='var(--accent-gradient)';this.style.color='#fff';this.style.borderColor='transparent';" onmouseout="this.style.background='none';this.style.color='var(--link-color)';this.style.borderColor='var(--border-color)';">Show more</button>
</div>

<script>
function toggleAwards() {
  var extras = document.querySelectorAll('.awards-extra');
  var btn = document.getElementById('awardsToggleBtn');
  var isHidden = extras[0] && extras[0].style.display === 'none';
  extras.forEach(function(el) {
    el.style.display = isHidden ? 'list-item' : 'none';
  });
  btn.textContent = isHidden ? 'Show less' : 'Show more';
  btn.style.borderColor = isHidden ? 'var(--accent)' : 'var(--border-color)';
}
</script>
