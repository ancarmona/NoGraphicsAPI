# OpenXR interop (Vulkan)

OpenXR's `XR_KHR_vulkan_enable2` makes the runtime create the Vulkan instance and device, choose the GPU, and own the
swapchain images. The Vulkan backend has four small additions for it. Metal has none of them: the calls fail or do nothing.

| Addition | Use |
| --- | --- |
| `DeviceDesc::vulkan_create_hooks` (`VulkanCreateHooks`) | `create_device` fills `VkInstanceCreateInfo` and `VkDeviceCreateInfo` as usual and passes them to the hooks instead of `vkCreateInstance` / `vkCreateDevice`. Forward them to `xrCreateVulkanInstanceKHR` and `xrCreateVulkanDeviceKHR`; the runtime adds its own extensions. `select_physical_device` returns the GPU of `xrGetVulkanGraphicsDevice2KHR`; that GPU must still have every capability the library needs. |
| `get_vulkan_handles` | The instance, physical device, device, and queue zero's family and index, for `XrGraphicsBindingVulkan2KHR`. |
| `create_external_texture` | Wraps a `VkImage` from `xrEnumerateSwapchainImages` as a `Texture`. `destroy_texture` leaves the image. |
| `begin_external_access` / `end_external_access` | The library keeps every image in `VK_IMAGE_LAYOUT_GENERAL`. After `xrWaitSwapchainImage`, `begin_external_access` moves the image from `UNDEFINED` (the contents are discarded) to `GENERAL`. Before `xrReleaseSwapchainImage`, `end_external_access` moves it to `COLOR_ATTACHMENT_OPTIMAL` (`DEPTH_STENCIL_ATTACHMENT_OPTIMAL` for depth): the layout the runtime requires. Both are ordered after every earlier command of the queue. |
| `RenderViewDesc::format` | A render view of a compatible format on a `mutable_format` texture. For an sRGB swapchain made with `XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT`, an `rgba8_unorm` view writes values that are already sRGB-encoded without encoding them again. The view's usage is limited to the attachment, so an image with storage usage still gets an sRGB-incompatible view. |

The OpenXR rule for the layouts (`XR_KHR_vulkan_enable2`): after `xrWaitSwapchainImage`, the image "has a memory layout
compatible with `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` for color images, or `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
for depth images", and "the application is responsible for transitioning the swapchain image back to the image layout and queue
ownership that the OpenXR runtime requires".

`tests/external_texture_test.cpp` plays the runtime: it makes the instance and device through the hooks and makes an image with
raw Vulkan. It clears the image through a UNORM view (0.6 stays 153, where an sRGB view would give 203). Then it copies the
image with a barrier from `COLOR_ATTACHMENT_OPTIMAL`, and the validation layer checks that layout.
