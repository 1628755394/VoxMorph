<script>
  import { onMount, onDestroy } from 'svelte';
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import {
    Activity,
    AudioLines,
    Cpu,
    Gauge,
    Mic,
    Palette,
    Play,
    Plug,
    Square,
    Volume2,
  } from 'lucide-svelte';

  // 背景图通过 Vite import 解析路径（避免 Svelte <style> 内 url 相对路径歧义）。
  import bgMesh from './assets/bg/mesh.svg';

  import { Button } from '$lib/components/ui/button';
  import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Separator } from '$lib/components/ui/separator';
  import { Slider } from '$lib/components/ui/slider';
  import { Progress } from '$lib/components/ui/progress';
  import { cn } from '$lib/utils';
  import {
    themePresets,
    defaultTheme,
    applyTheme,
    hslToHex,
    hexToHsl,
  } from '$lib/themes';

  // ─── Tauri 后端状态 ────────────────────────────────────────────
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

  // ─── 主题状态 ──────────────────────────────────────────────────
  let currentTheme = defaultTheme;
  let customPrimary = '#1a5fb4';
  let rootEl;

  $: presetList = Object.values(themePresets);

  function switchTheme(name) {
    currentTheme = name;
    const preset = themePresets[name];
    if (preset && rootEl) {
      applyTheme(rootEl, name);
      customPrimary = hslToHex(preset.vars['--primary']);
      localStorage.setItem('voxmorph-theme', name);
    }
  }

  function applyCustomColor(hex) {
    customPrimary = hex;
    if (!rootEl) return;
    const hsl = hexToHsl(hex);
    rootEl.style.setProperty('--primary', hsl);
    rootEl.style.setProperty('--accent', hsl);
    rootEl.style.setProperty('--ring', hsl);
    // 自定义色标记为 'custom'，避免与预设高亮冲突。
    currentTheme = 'custom';
    localStorage.setItem('voxmorph-theme', 'custom');
    localStorage.setItem('voxmorph-custom-color', hex);
  }

  // 从 localStorage 恢复主题（onMount 时调用）。
  function restoreTheme() {
    const saved = localStorage.getItem('voxmorph-theme');
    if (!saved || saved === 'custom') {
      const customColor = localStorage.getItem('voxmorph-custom-color');
      if (customColor) {
        applyCustomColor(customColor);
      } else {
        switchTheme(defaultTheme);
      }
    } else if (themePresets[saved]) {
      switchTheme(saved);
    } else {
      switchTheme(defaultTheme);
    }
  }

  // ─── Tauri 调用 ────────────────────────────────────────────────
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

  // 防抖推送实时参数到后端（滑块拖动时避免每像素都调命令）。
  let liveParamsTimer;
  async function updateLiveParams() {
    if (!isRunning()) return;
    clearTimeout(liveParamsTimer);
    liveParamsTimer = setTimeout(async () => {
      try {
        await invoke('set_live_params', {
          params: {
            pitch_shift: pitchShift,
            speaker_id: speakerId,
            input_gain: inputGain,
            output_gain: outputGain,
          }
        });
      } catch (e) {
        // 静默失败：参数更新失败不阻塞 UI（可能是 passthrough 模式无 handle）。
      }
    }, 80);
  }

  // 滑块值变化时自动推送。
  $: pitchShift, updateLiveParams();
  $: speakerId, updateLiveParams();
  $: inputGain, updateLiveParams();
  $: outputGain, updateLiveParams();

  // ─── 派生状态 ──────────────────────────────────────────────────
  const stateVariant = (s) => ({
    idle: 'secondary',
    'loading-model': 'warning',
    ready: 'success',
    running: 'default',
    error: 'destructive',
  }[s] ?? 'secondary');

  const stateLabel = (s) => ({
    idle: '空闲',
    'loading-model': '加载中',
    ready: '就绪',
    running: '运行中',
    error: '错误',
  }[s] ?? s);

  const isRunning = () => state === 'running';
  const isLoading = () => state === 'loading-model' || loading;
  const bufferPct = () => Math.min(100, (metrics.buffer_level / 8) * 100);
  const inferPct = () => Math.min(100, (metrics.infer_ms / 200) * 100);

  // ─── 生命周期 ──────────────────────────────────────────────────
  let unlistenState;
  let unlistenMetrics;
  let metricsInterval;

  onMount(async () => {
    restoreTheme();
    refreshState();
    refreshDevices();
    unlistenState = await listen('state-changed', (event) => {
      state = event.payload;
    });
    // 走事件通道接收指标（voxmorph skill：不用前端轮询 get_metrics）。
    unlistenMetrics = await listen('metrics-update', (event) => {
      metrics = event.payload;
      state = event.payload.state;
    });
    // 低频触发后端推送（后端 emit_metrics 主动推一次快照）。
    metricsInterval = setInterval(() => {
      invoke('emit_metrics').catch(() => {});
    }, 500);
  });

  onDestroy(() => {
    unlistenState?.();
    unlistenMetrics?.();
    clearInterval(metricsInterval);
  });
</script>

<div bind:this={rootEl} class="app-root">
  <!-- 背景层：渐变网格图（路径通过 Vite import 解析） -->
  <div class="bg-layer" aria-hidden="true" style="background-image: url('{bgMesh}')"></div>
  <!-- 玻璃质感遮罩层：双层模糊 + 暗化 -->
  <div class="glass-veil" aria-hidden="true"></div>

  <main class="content">
    <!-- 顶部：标题 + 主题切换 -->
    <header class="flex items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="logo-orb">
          <AudioLines class="h-5 w-5" />
        </div>
        <div>
          <h1 class="text-xl font-semibold tracking-tight">VoxMorph</h1>
          <p class="text-xs text-muted-foreground">AI 实时 / 离线变声器</p>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <Palette class="h-4 w-4 text-muted-foreground" />
        <div class="theme-swatches" role="group" aria-label="主题预设">
          {#each presetList as preset}
            <button
              class="swatch"
              class:active={currentTheme === preset.name}
              style={`background: hsl(${preset.vars['--primary']})`}
              aria-label={`切换到${preset.label}`}
              aria-pressed={currentTheme === preset.name}
              on:click={() => switchTheme(preset.name)}
            ></button>
          {/each}
        </div>
        <Separator orientation="vertical" className="h-6" />
        <label class="custom-color" aria-label="自定义主色">
          <input
            type="color"
            value={customPrimary}
            on:input={(e) => applyCustomColor(e.currentTarget.value)}
          />
          <span class="text-xs text-muted-foreground">自定义</span>
        </label>
      </div>
    </header>

    {#if error}
      <div class="error-banner" role="alert">
        {error}
      </div>
    {/if}

    <!-- 状态卡片 -->
    <Card className="glass-card">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle className="text-sm font-medium">引擎状态</CardTitle>
          <CardDescription className="mt-1">实时管线当前状态</CardDescription>
        </div>
        <Badge variant={stateVariant(state)} className="gap-1.5">
          <span class="status-dot" data-state={state}></span>
          {stateLabel(state)}
        </Badge>
      </CardHeader>
    </Card>

    <!-- RVC 模型配置 -->
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Cpu class="h-4 w-4" /> RVC 模型
        </CardTitle>
        <CardDescription>指定 ONNX 模型路径与采样参数</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 sm:grid-cols-2">
        <div class="grid gap-2">
          <Label for="embedder">ContentVec (embedder)</Label>
          <Input id="embedder" bind:value={embedderPath} disabled={isRunning()} placeholder="models/content_vec_500.onnx" />
        </div>
        <div class="grid gap-2">
          <Label for="f0">RMVPE (F0)</Label>
          <Input id="f0" bind:value={f0ModelPath} disabled={isRunning()} placeholder="models/rmvpe.onnx" />
        </div>
        <div class="grid gap-2">
          <Label for="rvc">RVC 模型</Label>
          <Input id="rvc" bind:value={rvcModelPath} disabled={isRunning()} placeholder="models/rvc.onnx" />
        </div>
        <div class="grid gap-2">
          <Label for="sr">RVC 采样率</Label>
          <Input id="sr" type="number" bind:value={rvcSampleRate} disabled={isRunning()} min="8000" max="96000" step="100" />
        </div>
        <div class="grid gap-2 sm:col-span-2">
          <Label for="ch">Embedder 通道数</Label>
          <Input id="ch" type="number" bind:value={embedderChannels} disabled={isRunning()} min="64" max="1024" step="64" />
        </div>
      </CardContent>
      <CardContent className="flex flex-wrap gap-2 pt-0">
        <Button on:click={startRvcEngine} disabled={isRunning() || isLoading()} className="gap-1.5">
          <Play class="h-4 w-4" /> 启动 RVC 变声
        </Button>
        <Button variant="outline" on:click={startPassthrough} disabled={isRunning() || isLoading()} className="gap-1.5">
          <Volume2 class="h-4 w-4" /> 直通模式
        </Button>
        <Button variant="destructive" on:click={stopEngine} disabled={!isRunning()} className="gap-1.5">
          <Square class="h-4 w-4" /> 停止引擎
        </Button>
      </CardContent>
    </Card>

    <!-- 实时参数 -->
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Gauge class="h-4 w-4" /> 实时参数
        </CardTitle>
        <CardDescription>变声过程中的可调参数</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5 sm:grid-cols-2">
        <div class="grid gap-2">
          <div class="flex items-center justify-between">
            <Label>Pitch shift（半音）</Label>
            <span class="text-sm font-medium text-primary tabular-nums">{pitchShift}</span>
          </div>
          <Slider bind:value={pitchShift} min={-24} max={24} step={1} disabled={isRunning()} />
        </div>
        <div class="grid gap-2">
          <div class="flex items-center justify-between">
            <Label>Speaker ID</Label>
            <span class="text-sm font-medium text-primary tabular-nums">{speakerId}</span>
          </div>
          <Slider bind:value={speakerId} min={0} max={100} step={1} disabled={isRunning()} />
        </div>
        <div class="grid gap-2">
          <div class="flex items-center justify-between">
            <Label>输入增益</Label>
            <span class="text-sm font-medium text-primary tabular-nums">{inputGain.toFixed(1)}×</span>
          </div>
          <Slider bind:value={inputGain} min={0} max={3} step={0.1} disabled={isRunning()} />
        </div>
        <div class="grid gap-2">
          <div class="flex items-center justify-between">
            <Label>输出增益</Label>
            <span class="text-sm font-medium text-primary tabular-nums">{outputGain.toFixed(1)}×</span>
          </div>
          <Slider bind:value={outputGain} min={0} max={3} step={0.1} disabled={isRunning()} />
        </div>
      </CardContent>
    </Card>

    <!-- 音频设备 -->
    <Card className="glass-card">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Mic class="h-4 w-4" /> 音频设备
          </CardTitle>
          <CardDescription className="mt-1">系统检测到的输入 / 输出设备</CardDescription>
        </div>
        <Button variant="ghost" size="sm" on:click={refreshDevices} disabled={loading} className="gap-1.5">
          <Plug class="h-4 w-4" /> 刷新
        </Button>
      </CardHeader>
      <CardContent>
        {#if loading}
          <div class="space-y-2" aria-busy="true" aria-label="枚举设备中">
            {#each Array(3) as _, i}
              <div class="h-10 animate-pulse rounded-md bg-muted/40"></div>
            {/each}
          </div>
        {:else if devices.length === 0}
          <div class="empty-state" role="status">
            <Mic class="mx-auto h-8 w-8 text-muted-foreground/50" />
            <p class="mt-2 text-sm text-muted-foreground">未检测到音频设备</p>
          </div>
        {:else}
          <ul role="list" class="divide-y divide-border/60">
            {#each devices as dev}
              <li class="flex items-center justify-between py-2.5">
                <span class="flex items-center gap-2 text-sm">
                  {dev.name}
                  {#if dev.is_default}
                    <Badge variant="success" className="text-[10px]">默认</Badge>
                  {/if}
                </span>
                <span class="text-xs text-muted-foreground tabular-nums">
                  {dev.sample_rate} Hz · {dev.channels} ch
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      </CardContent>
    </Card>

    <!-- 管线指标 -->
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <Activity class="h-4 w-4" /> 管线指标
        </CardTitle>
        <CardDescription>实时延迟与缓冲水位</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div class="grid grid-cols-3 gap-3">
          <div class="metric-cell">
            <div class="metric-value tabular-nums">{metrics.buffer_level}</div>
            <div class="metric-label">缓冲水位</div>
          </div>
          <div class="metric-cell">
            <div class="metric-value tabular-nums">{metrics.infer_ms.toFixed(1)}</div>
            <div class="metric-label">推理 (ms)</div>
          </div>
          <div class="metric-cell">
            <div class="metric-value tabular-nums">{metrics.dropped_frames}</div>
            <div class="metric-label">丢帧</div>
          </div>
        </div>
        <Separator />
        <div class="grid gap-3">
          <div class="grid gap-1.5">
            <div class="flex justify-between text-xs text-muted-foreground">
              <span>缓冲水位</span><span class="tabular-nums">{bufferPct().toFixed(0)}%</span>
            </div>
            <Progress value={bufferPct()} />
          </div>
          <div class="grid gap-1.5">
            <div class="flex justify-between text-xs text-muted-foreground">
              <span>推理耗时（相对 200ms 预算）</span><span class="tabular-nums">{inferPct().toFixed(0)}%</span>
            </div>
            <Progress value={inferPct()} />
          </div>
        </div>
      </CardContent>
    </Card>
  </main>
</div>

<style>
  /* 背景层：渐变网格 SVG，全屏覆盖 */
  .app-root {
    position: relative;
    min-height: 100vh;
    color: hsl(var(--foreground));
  }

  .bg-layer {
    position: fixed;
    inset: 0;
    background-size: cover;
    background-position: center;
    z-index: -2;
  }

  /* 玻璃质感遮罩：双层（暗化 + 模糊），不抢内容焦点 */
  .glass-veil {
    position: fixed;
    inset: 0;
    background:
      linear-gradient(180deg, hsl(var(--background) / 0.35) 0%, hsl(var(--background) / 0.65) 100%);
    backdrop-filter: blur(8px) saturate(140%);
    -webkit-backdrop-filter: blur(8px) saturate(140%);
    z-index: -1;
  }

  .content {
    max-width: 880px;
    margin: 0 auto;
    padding: 1.75rem 1.5rem 3rem;
  }

  /* 玻璃卡片：半透明 + 模糊 + 高光边 */
  :global(.glass-card) {
    background: hsl(var(--glass-bg, var(--card)));
    backdrop-filter: blur(hsl(var(--glass-blur, 16px))) saturate(160%);
    -webkit-backdrop-filter: blur(hsl(var(--glass-blur, 16px))) saturate(160%);
    border: 1px solid hsl(var(--glass-border, var(--border)));
    box-shadow:
      0 1px 0 0 hsl(0 0% 100% / 0.04) inset,
      0 8px 32px -12px hsl(0 0% 0% / 0.5);
    margin-bottom: 1rem;
  }

  /* Logo 球体 */
  .logo-orb {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 0.625rem;
    background: hsl(var(--primary) / 0.15);
    border: 1px solid hsl(var(--primary) / 0.3);
    color: hsl(var(--primary));
  }

  /* 主题色板 */
  .theme-swatches {
    display: flex;
    gap: 0.375rem;
  }
  .swatch {
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 9999px;
    border: 2px solid hsl(var(--border));
    cursor: pointer;
    transition: transform 0.15s, border-color 0.15s;
    padding: 0;
  }
  .swatch:hover { transform: scale(1.1); }
  .swatch.active {
    border-color: hsl(var(--foreground));
    box-shadow: 0 0 0 2px hsl(var(--background)), 0 0 0 3px hsl(var(--primary));
  }

  .custom-color {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    cursor: pointer;
  }
  .custom-color input[type="color"] {
    width: 1.25rem;
    height: 1.25rem;
    border: 2px solid hsl(var(--border));
    border-radius: 9999px;
    background: none;
    cursor: pointer;
    padding: 0;
    overflow: hidden;
  }
  .custom-color input[type="color"]::-webkit-color-swatch-wrapper { padding: 0; }
  .custom-color input[type="color"]::-webkit-color-swatch { border: none; border-radius: 9999px; }

  /* 错误横幅 */
  .error-banner {
    margin-top: 1rem;
    padding: 0.625rem 0.875rem;
    border-radius: 0.5rem;
    background: hsl(var(--destructive) / 0.12);
    border: 1px solid hsl(var(--destructive) / 0.3);
    color: hsl(var(--destructive));
    font-size: 0.85rem;
  }

  /* 状态点 */
  .status-dot {
    width: 0.4rem;
    height: 0.4rem;
    border-radius: 9999px;
    background: currentColor;
  }
  .status-dot[data-state="running"] { animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* 指标单元格 */
  .metric-cell {
    text-align: center;
    padding: 0.75rem;
    border-radius: 0.5rem;
    background: hsl(var(--muted) / 0.3);
    border: 1px solid hsl(var(--border) / 0.5);
  }
  .metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: hsl(var(--primary));
    line-height: 1.2;
  }
  .metric-label {
    margin-top: 0.25rem;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: hsl(var(--muted-foreground));
  }

  .empty-state {
    text-align: center;
    padding: 1.5rem 0;
  }

  /* 响应式：窄屏单列 */
  @media (max-width: 640px) {
    .content { padding: 1rem 0.875rem 2rem; }
    header { flex-direction: column; align-items: flex-start; gap: 0.75rem; }
  }
</style>
