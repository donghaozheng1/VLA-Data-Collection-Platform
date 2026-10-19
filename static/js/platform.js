/* 平台前端公共工具：接口调用、任务提交与轮询、格式化。 */

const API = {
  async get(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  async put(url, body) {
    const r = await fetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
};

function fmtBytes(n) {
  if (n === null || n === undefined) return '-';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB';
  return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return '-';
  if (seconds < 60) return seconds.toFixed(1) + 's';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m${s}s`;
}

function statusBadge(status) {
  const map = {
    succeeded: 'status-done', running: 'status-running', queued: 'status-paused',
    failed: 'status-error', ready: 'status-done',
  };
  const label = { succeeded: '成功', running: '执行中', queued: '排队中', failed: '失败' }[status] || status;
  return `<span class="status-badge ${map[status] || 'status-offline'}">${label}</span>`;
}

function severityBadge(sev) {
  const map = { error: 'status-error', warning: 'status-paused', info: 'status-running' };
  return `<span class="status-badge ${map[sev] || 'status-offline'}">${sev}</span>`;
}

function setMsg(el, text, kind) {
  const box = typeof el === 'string' ? document.getElementById(el) : el;
  if (!box) return;
  box.textContent = text;
  box.style.color = kind === 'error' ? '#ff4d4f' : (kind === 'ok' ? '#52c41a' : '#666');
}

/** 轮询任务直到结束，onTick(job) 每次都会拿到最新状态。 */
async function pollJob(jobId, onTick, intervalMs = 700) {
  for (;;) {
    const job = await API.get(`/api/jobs/${jobId}`);
    if (onTick) onTick(job);
    if (job.status === 'succeeded' || job.status === 'failed') return job;
    await new Promise((res) => setTimeout(res, intervalMs));
  }
}

/** 提交任务 + 进度条/文字自动更新，结束后回调 onDone(job)。 */
async function runJob(kind, payload, opts = {}) {
  const { barId, msgId, onDone } = opts;
  const bar = barId ? document.getElementById(barId) : null;
  const msg = msgId ? document.getElementById(msgId) : null;
  if (bar) { bar.style.display = 'block'; bar.style.width = '2%'; }
  setMsg(msg, '已提交，排队中…');
  try {
    const { job_id } = await API.post(`/api/jobs/${kind}`, payload);
    const job = await pollJob(job_id, (j) => {
      if (bar) bar.style.width = Math.max(2, j.progress) + '%';
      setMsg(msg, `[${j.kind_label}] ${j.message || j.status} ${j.progress}%`);
    });
    if (job.status === 'failed') {
      setMsg(msg, '失败：' + (job.error || '').split('\n')[0], 'error');
    } else {
      setMsg(msg, '完成', 'ok');
    }
    if (onDone) onDone(job);
    return job;
  } catch (err) {
    setMsg(msg, '调用出错：' + err.message, 'error');
    throw err;
  } finally {
    if (bar) setTimeout(() => { bar.style.display = 'none'; }, 1500);
  }
}

/** 任务列表（跨页面通用）。 */
async function renderJobs(tbodyId, limit = 12) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  const rows = await API.get(`/api/jobs?limit=${limit}`);
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#999;padding:18px;">还没有任务</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((j) => `<tr>
      <td>${j.id}</td>
      <td><span class="tag">${j.kind_label}</span></td>
      <td>${statusBadge(j.status)}</td>
      <td style="width:140px;">
        <div style="background:#e9edf4;border-radius:6px;height:8px;">
          <div style="background:#2b7de9;height:8px;border-radius:6px;width:${j.progress}%;"></div>
        </div>
      </td>
      <td style="font-size:13px;color:#666;">${j.message || '-'}</td>
      <td style="font-size:12px;color:#999;">${j.created_at || ''}</td>
      <td><button class="btn btn-sm" style="background:#f0f0f0;" onclick="showJob(${j.id})">详情</button></td>
    </tr>`).join('');
}

async function showJob(jobId) {
  const job = await API.get(`/api/jobs/${jobId}`);
  const lines = [
    `任务 #${job.id} · ${job.kind_label}`,
    `状态：${job.status}  进度：${job.progress}%`,
    `开始：${job.started_at || '-'}   结束：${job.finished_at || '-'}`,
    `消息：${job.message || '-'}`,
  ];
  if (job.error) lines.push('错误：' + job.error);
  if (job.result) lines.push('结果：' + JSON.stringify(job.result, null, 2));
  alert(lines.join('\n'));
}
