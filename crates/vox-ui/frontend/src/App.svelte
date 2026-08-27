<script>
  import { onMount } from 'svelte';
  import { invoke } from '@tauri-apps/api/core';

  let state = 'idle';
  let devices = [];
  let metrics = { buffer_level: 0, infer_ms: 0, dropped_frames: 0, state: 'idle' };
  let error = null;
  let loading = false;

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

  onMount(() => {
    refreshState();
    refreshDevices();
    // 轮询指标（M1 骨架；后续改为事件订阅）
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
</style>
