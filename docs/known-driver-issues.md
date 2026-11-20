# Known driver issues

These are locally reproduced observations, not vendor-confirmed root causes.

## NVIDIA 617.14: stale address-based texture readback

The failure observed on NVIDIA 596.99 persists on an RTX 4090 with NVIDIA 617.14, retested on 2026-10-07.
An upload through `vkCmdCopyMemoryToImageKHR` followed by `vkCmdCopyImageToMemoryKHR` returns incorrect
data despite a transfer-write to transfer-read barrier. Removing the intermediate timeline wait from
`test_texture_copy` fails for 3D, array, cube, and BC7 textures in both Release and Debug.
The buffer-command controls pass.

A parallel-recording control using two command buffers in one submission, with a transfer barrier and
no intermediate timeline wait, passed 50 runs in each Debug/Release address-command and buffer-command variant.
A timeline wait between separate upload and readback submissions works. The parallel texture and
pitch-roundtrip tests retain this workaround. No driver-specific barrier widening is applied by the graphics API.

## NVIDIA 621.24: a texture past 4 GiB of its memory is bound 4 GiB lower

Found on Windows with an RTX 4090 and NVIDIA 621.24 on 2026-11-20 (Surfbird T241). One `VkDeviceMemory` of more than 4 GiB
(`maxMemoryAllocationSize` is unlimited there), texture A bound at offset 0 and texture B, the same 1024 x 1024 `r32_uint`, bound at offset
4 GiB with `vkBindImageMemory` (a 64-bit offset). After an upload to A, then one to B, then readbacks, every texel of A holds B's data: B
uses the memory 4 GiB lower. The control, B ending exactly at 4 GiB, keeps both. In Surfbird's 4.8 GiB frame texture heap a texture
written by a compute pass at offset 4.18 GiB damaged the visibility buffer at 0.18 GiB.

The root cause is not vendor-confirmed; only offsets at and past 4 GiB were seen to fail. `create_texture` now refuses, with an error, a
texture whose bytes end past 4 GiB of its heap (a texture across 4 GiB too, as a margin), and returns null; `test_texture_heap_4g` checks
the control and the refusals. An application with more than 4 GiB of textures uses several texture heaps.

## NVIDIA 596.99 to 621.24: concurrent submissions to several queue families lose the device

The failure observed on NVIDIA 596.99 persists on Windows with an RTX 4090 and NVIDIA 617.14 and 621.24.
Buffer copies submitted to separate general, compute, and copy queues from three CPU threads
intermittently return `VK_ERROR_DEVICE_LOST`.

The failure reproduces independently of NoGraphicsAPI using standard Vulkan 1.3, with no instance or
device extensions enabled (`--buffers --no-validation`). The standalone repro does not link or call
NoGraphicsAPI; it uses ordinary `vkCmdCopyBuffer2`, synchronization2 barriers, and timeline semaphores.
There are no shaders, PushData calls, descriptor heaps, or timestamp queries.

The 2026-10-07 retest reproduced core-copy device loss without validation in both Release and Debug.
Address copies also failed with and without synchronization validation. Core-copy validation runs and
sequential address-copy runs passed 50 repetitions each; these controls do not establish a workaround.

This points to an NVIDIA driver issue independent of the library implementation, not a confirmed
NoGraphicsAPI defect. The root cause is not vendor-confirmed. The 2026-11-16 retest on 621.24 narrowed it: two threads calling
`vkQueueSubmit2` at the same time on the general and compute families are enough, and a lock around that call alone stops it.
The Vulkan backend now takes such a lock on devices with several queue families (`NGA_SUBMIT_LOCK=0` turns it off).
See the [standalone repro, test results and workaround](repro-queue-device-lost.md).

## Metal 4 / macOS 26.6.2: render timestamps leave counter entries unwritten

On Apple M3 Max with macOS 26.6.2 and Xcode 27, ten native render timestamp writes interleaved with
draws return five nonzero entries and five zeros after GPU completion. A standalone native Metal repro
fails with and without API validation, using one ordinary render pass and no Slang or NoGraphicsAPI calls.

This points to a Metal driver/runtime issue; the root cause is not vendor-confirmed. Outside-render
markers work in the tested configuration, but no workaround for in-render markers is established.
`test_render_continuation` retains the failing check. See the [repro and results](repro-metal-render-timestamps.md).

## MetalTools validation limitations

MetalTools has separate limitations with concurrent sampler lifetime changes, placed-resource
residency, and indirect mesh instrumentation. See [Metal validation](metal-validation.md#metaltools-limitations)
for reproductions and tool-specific handling.
