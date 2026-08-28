// VoxMorph 主题预设。每套配色用 HSL 通道值（与 shadcn token 一致）。
// 切换主题只需把对应预设的 CSS 变量写到 :root 或 .theme-xxx 上。
//
// 设计原则（遵循 frontend-ui-engineering skill）：
// - 不用 AI 默认的紫蓝渐变，每套都有明确语义
// - 主色对比度 ≥ 4.5:1（AA）
// - 玻璃质感层用 token 透明度派生，不硬编码

/** @typedef {{ name: string, label: string, vars: Record<string, string> }} ThemePreset */

/** @type {Record<string, ThemePreset>} */
export const themePresets = {
  midnight: {
    name: 'midnight',
    label: '午夜蓝',
    vars: {
      '--background': '230 30% 8%',
      '--foreground': '220 13% 87%',
      '--card': '224 41% 16%',
      '--card-foreground': '220 13% 87%',
      '--popover': '224 41% 16%',
      '--popover-foreground': '220 13% 87%',
      '--primary': '213 78% 40%',
      '--primary-foreground': '0 0% 100%',
      '--secondary': '224 30% 20%',
      '--secondary-foreground': '220 13% 87%',
      '--muted': '224 30% 20%',
      '--muted-foreground': '233 9% 60%',
      '--accent': '213 78% 40%',
      '--accent-foreground': '0 0% 100%',
      '--destructive': '0 72% 56%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '233 27% 23%',
      '--input': '233 27% 23%',
      '--ring': '213 78% 40%',
      // 玻璃层派生：卡片半透明 + 模糊强度
      '--glass-bg': '224 41% 16% / 0.55',
      '--glass-border': '220 30% 90% / 0.08',
      '--glass-blur': '16px',
    },
  },
  ember: {
    name: 'ember',
    label: '余烬橙',
    vars: {
      '--background': '20 14% 8%',
      '--foreground': '36 33% 90%',
      '--card': '24 18% 14%',
      '--card-foreground': '36 33% 90%',
      '--popover': '24 18% 14%',
      '--popover-foreground': '36 33% 90%',
      '--primary': '24 89% 56%',
      '--primary-foreground': '20 14% 8%',
      '--secondary': '24 14% 20%',
      '--secondary-foreground': '36 33% 90%',
      '--muted': '24 14% 20%',
      '--muted-foreground': '28 10% 62%',
      '--accent': '24 89% 56%',
      '--accent-foreground': '20 14% 8%',
      '--destructive': '0 72% 56%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '24 14% 22%',
      '--input': '24 14% 22%',
      '--ring': '24 89% 56%',
      '--glass-bg': '24 18% 14% / 0.55',
      '--glass-border': '40 40% 90% / 0.08',
      '--glass-blur': '16px',
    },
  },
  forest: {
    name: 'forest',
    label: '深林绿',
    vars: {
      '--background': '160 20% 7%',
      '--foreground': '150 18% 88%',
      '--card': '158 22% 13%',
      '--card-foreground': '150 18% 88%',
      '--popover': '158 22% 13%',
      '--popover-foreground': '150 18% 88%',
      '--primary': '152 56% 42%',
      '--primary-foreground': '160 20% 7%',
      '--secondary': '158 14% 20%',
      '--secondary-foreground': '150 18% 88%',
      '--muted': '158 14% 20%',
      '--muted-foreground': '155 10% 60%',
      '--accent': '152 56% 42%',
      '--accent-foreground': '160 20% 7%',
      '--destructive': '0 72% 56%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '158 14% 22%',
      '--input': '158 14% 22%',
      '--ring': '152 56% 42%',
      '--glass-bg': '158 22% 13% / 0.55',
      '--glass-border': '150 40% 90% / 0.08',
      '--glass-blur': '16px',
    },
  },
  rose: {
    name: 'rose',
    label: '玫瑰粉',
    vars: {
      '--background': '340 22% 8%',
      '--foreground': '340 18% 90%',
      '--card': '338 26% 14%',
      '--card-foreground': '340 18% 90%',
      '--popover': '338 26% 14%',
      '--popover-foreground': '340 18% 90%',
      '--primary': '340 75% 55%',
      '--primary-foreground': '340 22% 8%',
      '--secondary': '338 18% 20%',
      '--secondary-foreground': '340 18% 90%',
      '--muted': '338 18% 20%',
      '--muted-foreground': '340 10% 62%',
      '--accent': '340 75% 55%',
      '--accent-foreground': '340 22% 8%',
      '--destructive': '0 72% 56%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '338 18% 22%',
      '--input': '338 18% 22%',
      '--ring': '340 75% 55%',
      '--glass-bg': '338 26% 14% / 0.55',
      '--glass-border': '340 40% 90% / 0.08',
      '--glass-blur': '16px',
    },
  },
  mono: {
    name: 'mono',
    label: '极简灰',
    vars: {
      '--background': '220 10% 8%',
      '--foreground': '220 10% 88%',
      '--card': '220 10% 13%',
      '--card-foreground': '220 10% 88%',
      '--popover': '220 10% 13%',
      '--popover-foreground': '220 10% 88%',
      '--primary': '220 10% 75%',
      '--primary-foreground': '220 10% 8%',
      '--secondary': '220 8% 20%',
      '--secondary-foreground': '220 10% 88%',
      '--muted': '220 8% 20%',
      '--muted-foreground': '220 8% 60%',
      '--accent': '220 10% 75%',
      '--accent-foreground': '220 10% 8%',
      '--destructive': '0 72% 56%',
      '--destructive-foreground': '0 0% 100%',
      '--border': '220 8% 22%',
      '--input': '220 8% 22%',
      '--ring': '220 10% 75%',
      '--glass-bg': '220 10% 13% / 0.55',
      '--glass-border': '220 20% 90% / 0.08',
      '--glass-blur': '16px',
    },
  },
};

/** 默认主题。 */
export const defaultTheme = 'midnight';

/** 把预设的 CSS 变量写到目标元素上。 */
export function applyTheme(target, presetName) {
  const preset = themePresets[presetName] ?? themePresets[defaultTheme];
  for (const [k, v] of Object.entries(preset.vars)) {
    target.style.setProperty(k, v);
  }
}

/** 把 HSL 字符串转成可被 input[type=color] 接受的 hex（取主色）。 */
export function hslToHex(hsl) {
  const [h, s, l] = hsl.match(/\d+(\.\d+)?/g).map(Number);
  const lt = l / 100;
  const a = (s * Math.min(lt, 1 - lt)) / 100;
  const f = (n) => {
    const k = (n + h / 30) % 12;
    const c = lt - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * c)
      .toString(16)
      .padStart(2, '0');
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

/** hex 转 HSL 通道字符串（"H S% L%"）。 */
export function hexToHsl(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      default: h = (r - g) / d + 4;
    }
    h *= 60;
  }
  return `${Math.round(h)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`;
}
