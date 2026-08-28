<script>
  import { Progress as ProgressPrimitive } from 'bits-ui';
  import { cn } from '$lib/utils';

  export let value = 0;
  export let max = 100;
  export let className = undefined;
</script>

<ProgressPrimitive.Root
  {value}
  {max}
  class={cn('relative h-2 w-full overflow-hidden rounded-full bg-primary/20', className)}
  {...$$restProps}
>
  <svelte:fragment let:builder>
    <div
      class="h-full bg-primary transition-all"
      style={`width: ${Math.min(100, (value / max) * 100)}%`}
      aria-hidden="true"
    ></div>
  </svelte:fragment>
</ProgressPrimitive.Root>
