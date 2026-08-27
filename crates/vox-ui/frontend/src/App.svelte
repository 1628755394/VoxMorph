<script>
  import { onMount } from 'svelte';
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';

  let state = 'idle';
  let devices = [];
  let metrics = { buffer_level: 0, infer_ms: 0, dropped_frames: 0, state: 'idle' };
  let error = null;
  let loading = false;

  // RVC 模型路径
  let embedderPath = 'models/content_vec_500.onnx';
  let f0ModelPath = 'models/rmvpe.onnx';
  let rvcModelPath = 'models/rvc.onnx';
  let rvcSampleRate = 48000;
  let embedderChannels = 256;

  // 实时参数
  let pitchShift = 0;
  let speakerId = 0;
  let inputGain = 1.0;
  let outputGain = 1.0;

  async function refreshState() {
    try {
      state = await invoke('get_state');
      error = null;
    } catch (e) {
      error = String(e);
    }
  }

  async function refreshDevices() {
    loading = true;
    try {
      devices = await invoke('list_audio_devices');
      error = null;
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function refreshMetrics() {
    try {
      metrics = await invoke('get_metrics');
      state = metrics.state;
      error = null;
    } catch (e) {
      error = String(e);
    }
  }

  async function startRvcEngine() {
    loading = true;
    error = null;
    try {
      await invoke('start_rvc_engine', {
        paths: {
          embedder: embedderPath,
          f0_model: f0ModelPath,
          rvc_model: rvcModelPath,
          rvc_sample_rate: rvcSampleRate,
          embedder_channels: embedderChannels,
        }
      });
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
      refreshState();
    }
  }

  async function startPassthrough() {
    loading = true;
    error = null;
    try {
      await invoke('start_engine');
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
      refreshState();
    }
  }

  async function stopEngine() {
    loading = true;
    error = null;
    try {
      await invoke('stop_engine');
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
      refreshState();
    }
  }

  async function updateLiveParams() {
    // TODO: 后端需要 set_live_params 命令
    // 当前仅本地状态，后续接入
  }

  const isRunning = () => state === 'running';
  const isLoading = () => state === 'loading-model' || loading;

  onMount(() => {
    refreshState();
    refreshDevices();
    // 监听状态变化事件
    listen('state-changed', (event) => {
      state = event.payload;
    });
    // 轮询指标
    const interval = setInterval(refreshMetrics, 1000);
    return () => clearInterval(interval);
  });
</script>

<header>
  <h1>VoxMorph</h1>
  <p class="subtitle">AI 实时/离线变声器</p>
</header>

{#if error}
  <div class="error-msg">{error}</div>
{/if}

<section class="card">
  <h2>状态</h2>
  <span class="state-badge {state}">{state}</span>
</section>

<section class="card">
  <h2>RVC 模型</h2>
  <div class="form-grid">
    <label class="form-label">
      <span>ContentVec (embedder)</span>
      <input type="text" bind:value={embedderPath} disabled={isRunning()} placeholder="models/content_vec_500.onnx" />
    </label>
    <label class="form-label">
      <span>RMVPE (F0)</span>
      <input type="text" bind:value={f0ModelPath} disabled={isRunning()} placeholder="models/rmvpe.onnx" />
    </label>
    <label class="form-label">
      <span>RVC 模型</span>
      <input type="text" bind:value={rvcModelPath} disabled={isRunning()} placeholder="models/rvc.onnx" />
    </label>
    <label class="form-label">
      <span>RVC 采样率</span>
      <input type="number" bind:value={rvcSampleRate} disabled={isRunning()} min="8000" max="96000" step="100" />
    </label>
    <label class="form-label">
      <span>Embedder 通道数</span>
      <input type="number" bind:value={embedderChannels} disabled={isRunning()} min="64" max="1024" step="64" />
    </label>
  </div>
  <div class="button-row">
    <button on:click={startRvcEngine} disabled={isRunning() || isLoading()}>
      启动 RVC 变声
    </button>
    <button on:click={startPassthrough} disabled={isRunning() || isLoading()}>
      直通模式
    </button>
    <button on:click={stopEngine} disabled={!isRunning()} class="danger">
      停止引擎
    </button>
  </div>
</section>

<section class="card">
  <h2>实时参数</h2>
  <div class="form-grid">
    <label class="form-label">
      <span>Pitch shift (半音)</span>
      <input type="range" bind:value={pitchShift} min="-24" max="24" step="1" on:input={updateLiveParams} />
      <span class="range-value">{pitchShift}</span>
    </label>
    <label class="form-label">
      <span>Speaker ID</span>
      <input type="number" bind:value={speakerId} min="0" max="100" step="1" on:input={updateLiveParams} />
    </label>
    <label class="form-label">
      <span>输入增益</span>
      <input type="range" bind:value={inputGain} min="0" max="3" step="0.1" on:input={updateLiveParams} />
      <span class="range-value">{inputGain.toFixed(1)}</span>
    </label>
    <label class="form-label">
      <span>输出增益</span>
      <input type="range" bind:value={outputGain} min="0" max="3" step="0.1" on:input={updateLiveParams} />
      <span class="range-value">{outputGain.toFixed(1)}</span>
    </label>
  </div>
</section>

<section class="card">
  <h2>音频设备</h2>
  {#if loading}
    <p class="device-meta">枚举中...</p>
  {:else if devices.length === 0}
    <p class="device-meta">未检测到音频设备</p>
  {:else}
    <ul class="device-list">
      {#each devices as dev}
        <li class="device-item">
          <span class="device-name">
            {dev.name}
            {#if dev.is_default}<span class="device-default">★ 默认</span>{/if}
          </span>
          <span class="device-meta">
            {dev.sample_rate} Hz · {dev.channels} ch
          </span>
        </li>
      {/each}
    </ul>
  {/if}
  <button on:click={refreshDevices} style="margin-top: 0.75rem;">
    刷新设备列表
  </button>
</section>

<section class="card">
  <h2>管线指标</h2>
  <div class="metrics-grid">
    <div class="metric">
      <div class="metric-value">{metrics.buffer_level}</div>
      <div class="metric-label">缓冲水位</div>
    </div>
    <div class="metric">
      <div class="metric-value">{metrics.infer_ms.toFixed(1)}</div>
      <div class="metric-label">推理 (ms)</div>
    </div>
    <div class="metric">
      <div class="metric-value">{metrics.dropped_frames}</div>
      <div class="metric-label">丢帧</div>
    </div>
  </div>
</section>

<style>
  header {
    margin-bottom: 1.5rem;
  }
  h1 {
    font-size: 1.75rem;
    color: var(--accent-bright);
  }
  .subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
  }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
  }

  .form-label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .form-label > span:first-child {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .form-label input[type="text"],
  .form-label input[type="number"] {
    padding: 0.4rem 0.6rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg);
    color: var(--text);
    font-size: 0.85rem;
  }

  .form-label input[type="range"] {
    width: 100%;
  }

  .range-value {
    font-size: 0.8rem;
    color: var(--accent-bright);
    text-align: right;
  }

  .button-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .button-row button.danger {
    border-color: var(--error);
  }

  .button-row button.danger:hover:not(:disabled) {
    background: var(--error);
  }
</style>
