<script>
  import { cn } from '$lib/utils';

  export let className = undefined;
</script>

<h3 class={cn('text-lg font-semibold leading-none tracking-tight', className)} {...$$restProps}>
  <slot />
</h3>
