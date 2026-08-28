<script>
  import { Slider as SliderPrimitive } from 'bits-ui';
  import { cn } from '$lib/utils';

  export let value = 0;
  export let min = 0;
  export let max = 100;
  export let step = 1;
  export let disabled = false;
  export let className = undefined;
</script>

<SliderPrimitive.Root
  bind:value
  {min}
  {max}
  {step}
  {disabled}
  class={cn('relative flex w-full touch-none select-none items-center', className)}
  {...$$restProps}
>
  <svelte:fragment let:thumbs>
    <span class="relative h-1.5 w-full grow overflow-hidden rounded-full bg-primary/20">
      <span class="absolute h-full bg-primary" style={`right: ${(1 - (value - min) / (max - min)) * 100}%`} />
    </span>
    {#each thumbs as thumb}
      <span
        {...thumb}
        class="block h-4 w-4 rounded-full border border-primary/50 bg-background shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
      />
    {/each}
  </svelte:fragment>
</SliderPrimitive.Root>
