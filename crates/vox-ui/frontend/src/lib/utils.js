import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** 合并 Tailwind class，处理冲突与条件类。 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
