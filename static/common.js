const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const IS_ADMIN = location.pathname.startsWith('/admin');
const KEY = IS_ADMIN ? 'admin_token' : 'user_token';
const HOME = IS_ADMIN ? '/admin' : '/';

async function api(path, {method = 'GET', json, form, noRedirect} = {}) {
  const headers = {}, t = localStorage.getItem(KEY);
  if (t) headers.Authorization = 'Bearer ' + t;
  let body;
  if (json) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(json); }
  else if (form) body = form;
  const r = await fetch('/api' + path, {method, headers, body});
  if (r.status === 401 && !noRedirect) { localStorage.removeItem(KEY); location.href = HOME; throw new Error('Session expired'); }
  if (!r.ok) {
    let m = 'Request failed';
    try { const j = await r.json(); m = typeof j.detail === 'string' ? j.detail : 'Please check your input.'; } catch {}
    throw new Error(m);
  }
  return r.json();
}
async function blobUrl(path) {
  const r = await fetch('/api' + path, {headers: {Authorization: 'Bearer ' + localStorage.getItem(KEY)}});
  return r.ok ? URL.createObjectURL(await r.blob()) : null;
}
function toast(msg, bad) {
  const t = document.createElement('div'); t.className = 'toast' + (bad ? ' bad' : ''); t.textContent = msg;
  t.setAttribute('role', 'status'); document.body.append(t); setTimeout(() => t.remove(), 3500);
}
const fmtDate = s => new Date(s).toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'});
const pct = x => (x * 100).toFixed(1) + '%';
const nice = l => ({NORMAL: 'Normal', PNEUMONIA: 'Pneumonia', NOT_A_XRAY: 'Not a chest X-ray'}[l] ||
  String(l).toLowerCase().replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase()));
function verdict(a) {
  if (a.status === 'rejected') return {text: 'Rejected', cls: 'bad'};
  if (a.status === 'needs_review') return {text: 'Needs review', cls: 'warn'};
  return {text: nice(a.label), cls: a.label === 'NORMAL' ? 'good' : 'warn'};
}
function bindMenu() {
  const sb = $('#sidebar'), mb = $('#menuBtn');
  mb && (mb.onclick = () => sb.classList.toggle('open'));
  sb && sb.addEventListener('click', e => { if (e.target.closest('button')) sb.classList.remove('open'); });
}
function renderResult(a, imgSrc) {
  const v = verdict(a), d = a.details || {};
  const probs = Object.entries(a.probs || {}).sort((x, y) => y[1] - x[1]).map(([k, p]) =>
    `<div class="small row sp"><span>${esc(nice(k))}</span><span>${pct(p)}</span></div><div class="bar"><i style="width:${p * 100}%"></i></div>`).join('');
  const frames = d.frames ? `<h3 style="margin-top:1rem">Frames analysed (${d.analysed} of ${d.sampled} sampled)</h3>
    <p class="small muted">Skipped: ${d.skipped.not_xray} not X-ray-like, ${d.skipped.duplicate} near-duplicates. ${esc(d.rule)}</p>
    <div class="scroll"><table><tr><th>Time</th><th>Result</th><th>Confidence</th></tr>${d.frames.map(f =>
      `<tr><td>${f.t}s</td><td>${esc(nice(f.label))}</td><td>${pct(f.confidence)}</td></tr>`).join('')}</table></div>` : '';
  return `<div class="result">
    ${d.demo ? '<div class="banner">Demo mode: no trained model is loaded, so this result is a placeholder and not a real prediction.</div>' : ''}
    <div class="result-grid">
      <div>${imgSrc ? `<img src="${imgSrc}" alt="Analysed chest X-ray">` : '<p class="muted small">No image stored for this record.</p>'}</div>
      <div>
        <div class="row"><span class="pill ${v.cls}">${esc(v.text)}</span>
          ${a.status !== 'rejected' ? `<span class="muted">Model confidence estimate: ${pct(a.confidence)}</span>` : ''}</div>
        ${a.status === 'rejected' ? `<p>${esc(d.note || 'This file was not accepted.')}</p>` : ''}
        ${a.status === 'needs_review' ? `<p class="small">The model is not confident enough to give a clear answer${d.agreement !== undefined ? ' or the frames disagree' : ''}. Have a clinician review the original image.</p>` : ''}
        <div style="margin-top:.8rem">${probs}</div>
        <p class="small muted">Model ${esc(a.model_version)} · ${fmtDate(a.created)}</p>
      </div></div>${frames}
    <div class="disclaimer">${esc(d.disclaimer || 'Educational prototype. Not a medical diagnosis.')} Confidence is a model estimate, not the probability that you have pneumonia.</div></div>`;
}
