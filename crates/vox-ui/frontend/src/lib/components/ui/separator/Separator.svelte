<script>
  import { Separator as SeparatorPrimitive } from 'bits-ui';
  import { cn } from '$lib/utils';

  export let orientation = 'horizontal';
  export let className = undefined;
</script>

<SeparatorPrimitive.Root
  {orientation}
  class={cn(
    'shrink-0 bg-border',
    orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
    className
  )}
  {...$$restProps}
/>
