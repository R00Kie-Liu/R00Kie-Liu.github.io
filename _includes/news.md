<h2 id="news" style="margin: 60px 0px 10px;">News</h2>

<ul class="news-list">
<li><strong>[07/2026]</strong> Our team releases <a href="https://huggingface.co/inclusionAI/LLaDA2.2-flash"><strong>LLaDA 2.2</strong></a>, enabling agentic diffusion language models via Levenshtein editing.</li>
<li><strong>[02/2026]</strong> 😁 Paper about <a href="./pub_img/vim.txt">Few-shot action recogition</a> is accepted to <strong>TPAMI</strong>.</li>
<li><strong>[02/2026]</strong> ✍🏻 Our team release <strong>LLaDA 2.1</strong>, which supports token editing now!</li>
<li><strong>[12/2025]</strong> 🤖 Our team release the largest diffusion large language model (dLLM) <strong>LLaDA 2.0</strong></li>
<li><strong>[11/2025]</strong> 🥳 Paper about <a href="./pub_img/cogstream.txt">Streaming Video Reasoning</a> is accepted to <strong>AAAI</strong>.</li>
<li><strong>[06/2025]</strong> 👨‍🎓 I successfully defended my <strong>Ph.D.</strong> dissertation at SJTU！</li>
<li class="news-extra" style="display:none;"><strong>[06/2025]</strong> 😊 Paper about <a href="./pub_img/CSTA.txt">video recognition</a> is accepted to <strong>TCSVT</strong>.</li>
<li class="news-extra" style="display:none;"><strong>[02/2025]</strong> 😭 Paper about <a href="./pub_img/entailment.txt">video QA & reasoning</a> is accepted to <strong>CVPR 2025</strong>, many thanks to Filip and Cees!</li>
<li class="news-extra" style="display:none;"><strong>[09/2024]</strong> 😊 Paper about <a href="./pub_img/mecd.txt">video causal discovery & reasoning</a> is accepted to <strong>NeurIPS 2024</strong>.</li>
  <li class="news-extra" style="display:none;"><strong>[07/2024]</strong> 🥳 Paper about <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00720.pdf">video QA & reasoning</a> is accepted to <strong>ECCV 2024</strong>.</li>
  <li class="news-extra" style="display:none;"><strong>[03/2024]</strong> 😃 Paper about <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Wu_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_CVPR_2024_paper.html
">dense video captioning</a> is accepted to <strong>CVPR 2024</strong>.</li>
   <li class="news-extra" style="display:none;"><strong>[01/2024]</strong> 🛫 Start my visit to the <a href="https://ivi.fnwi.uva.nl/vislab/">VISLab</a> at University of Amsterdam as a <strong>visiting researcher</strong>.</li>
  <li class="news-extra" style="display:none;"><strong>[12/2023]</strong> 😆 Paper about <a href="https://ojs.aaai.org/index.php/AAAI/article/view/27983
">instructional video understanding</a> is accepted to <strong>AAAI 2024</strong>.</li>
   <li class="news-extra" style="display:none;"><strong>[07/2023]</strong> 😚 Paper about <a href="https://link.springer.com/article/10.1007/s11263-023-01842-6
">human-centric video analysis</a> is accepted to  <strong>IJCV</strong> journal.</li>
  <li class="news-extra" style="display:none;"><strong>[07/2022]</strong> 😁 Paper about <a href="https://arxiv.org/pdf/2207.09759">few-shot action recognition</a> is accepted to <strong>ACM MM 2022</strong>.</li>
  <li class="news-extra" style="display:none;"><strong>[11/2021]</strong> 😊 Paper about <a href="https://ojs.aaai.org/index.php/AAAI/article/view/20029/19788">few-shot action recognition</a> is accepted to <strong>AAAI 2022</strong>.</li>
   <li class="news-extra" style="display:none;"><strong>[05/2021]</strong> 🥳 Paper about <a href="https://ieeexplore.ieee.org/abstract/document/9459475/">fine-grained image recognition</a> is accepted to <strong>TMM</strong> journal.</li>
  <li class="news-extra" style="display:none;"><strong>[12/2020]</strong> 🏆 Our hosted <a href="http://humaninevents.org/">HiEve Challenge</a> won the <strong>best Challenge organization award</strong> in ACM MM 2020.</li>
</ul>

<div class="news-toggle" style="text-align:center; margin-top: 8px;">
  <button onclick="toggleNews()" id="newsToggleBtn" style="
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
function toggleNews() {
  var extras = document.querySelectorAll('.news-extra');
  var btn = document.getElementById('newsToggleBtn');
  var isHidden = extras[0] && extras[0].style.display === 'none';
  extras.forEach(function(el) {
    el.style.display = isHidden ? 'list-item' : 'none';
  });
  btn.textContent = isHidden ? 'Show less' : 'Show more';
  btn.style.borderColor = isHidden ? 'var(--accent)' : 'var(--border-color)';
}
</script>
