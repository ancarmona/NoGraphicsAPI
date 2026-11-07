#include <NoGraphicsAPI/NoGraphicsAPI.hpp>

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <string.h>

// The OpenXR interop of the Vulkan backend, without a runtime: the test plays the runtime's part.
// - VulkanCreateHooks: the device's instance and device come from the hooks (as xrCreateVulkanInstanceKHR and
//   xrCreateVulkanDeviceKHR would make them), on the GPU the hook selects; get_vulkan_handles gives them back.
// - create_external_texture: a 2-layer RGBA8 sRGB image made with raw Vulkan (MUTABLE_FORMAT), as a swapchain image.
// - A render view of format rgba8_unorm writes the cleared value without sRGB encoding (0.6 stays 153, not 203).
// - begin_external_access / end_external_access: the image ends in COLOR_ATTACHMENT_OPTIMAL. The "runtime" then records a
//   barrier from that layout and copies the image to a buffer; the validation layer checks the layout it expects.
namespace
{
constexpr int skipped = 77;
constexpr uint32 size = 16;
constexpr uint32 layers = 2;

struct HookLog
{
    int instance_calls = 0;
    int select_calls = 0;
    int device_calls = 0;
    VkPhysicalDevice selected = VK_NULL_HANDLE;
};

int32 create_instance(void* user, const void* create_info, void* instance) noexcept
{
    static_cast<HookLog*>(user)->instance_calls++;
    return vkCreateInstance(static_cast<const VkInstanceCreateInfo*>(create_info), nullptr, static_cast<VkInstance*>(instance));
}

// The first discrete GPU, else the first one: what a runtime driving a headset on that GPU would return
void* select_physical_device(void* user, void* instance) noexcept
{
    HookLog& log = *static_cast<HookLog*>(user);
    log.select_calls++;
    VkPhysicalDevice devices[16]{};
    uint32 count = 16;
    if (vkEnumeratePhysicalDevices(static_cast<VkInstance>(instance), &count, devices) < VK_SUCCESS || count == 0)
        return nullptr;
    log.selected = devices[0];
    for (uint32 index = 0; index < count; ++index)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(devices[index], &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            log.selected = devices[index];
            break;
        }
    }
    return log.selected;
}

int32 create_device(void* user, void* physical_device, const void* create_info, void* device) noexcept
{
    static_cast<HookLog*>(user)->device_calls++;
    return vkCreateDevice(static_cast<VkPhysicalDevice>(physical_device), static_cast<const VkDeviceCreateInfo*>(create_info), nullptr,
                          static_cast<VkDevice*>(device));
}

bool find_memory(VkPhysicalDevice physical_device, uint32 bits, VkMemoryPropertyFlags flags, uint32& output) noexcept
{
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (uint32 index = 0; index < properties.memoryTypeCount; ++index)
    {
        if ((bits & (1u << index)) && (properties.memoryTypes[index].propertyFlags & flags) == flags)
        {
            output = index;
            return true;
        }
    }
    return false;
}

int fail(const char* message) noexcept
{
    fprintf(stderr, "external texture: %s\n", message);
    return 1;
}
}

int main()
{
    HookLog log;
    const gpu::VulkanCreateHooks hooks{
        .user = &log,
        .create_instance = create_instance,
        .select_physical_device = select_physical_device,
        .create_device = create_device,
    };
    const gpu::DeviceInit init = gpu::create_device({.desired_queue_count = 1, .timestamp_query_count = 0, .vulkan_create_hooks = &hooks});
    if (init.error == gpu::Error::unsupported)
        return skipped;
    if (!init.device)
        return fail("create_device with hooks failed");
    gpu::Device* device = init.device;
    if (log.instance_calls != 1 || log.select_calls != 1 || log.device_calls != 1)
        return fail("each hook must be called once");
    const gpu::VulkanHandles handles = gpu::get_vulkan_handles(device);
    const VkDevice vk_device = static_cast<VkDevice>(handles.device);
    const VkPhysicalDevice vk_physical = static_cast<VkPhysicalDevice>(handles.physical_device);
    if (!handles.instance || !vk_device || vk_physical != log.selected || handles.queue_index != 0)
        return fail("get_vulkan_handles must give the hooks' objects and the selected GPU");

    // The "swapchain image": made outside the library, as a runtime would
    const VkFormat view_formats[] = {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM};
    const VkImageFormatListCreateInfo format_list{
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
        .viewFormatCount = 2,
        .pViewFormats = view_formats,
    };
    const VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &format_list,
        .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .extent = {size, size, 1},
        .mipLevels = 1,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    VkMemoryRequirements requirements{};
    uint32 memory_type = 0;
    if (vkCreateImage(vk_device, &image_info, nullptr, &image) != VK_SUCCESS)
        return fail("vkCreateImage");
    vkGetImageMemoryRequirements(vk_device, image, &requirements);
    if (!find_memory(vk_physical, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_type))
        return fail("no device-local memory for the image");
    const VkMemoryAllocateInfo image_allocation{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memory_type,
    };
    if (vkAllocateMemory(vk_device, &image_allocation, nullptr, &image_memory) != VK_SUCCESS ||
        vkBindImageMemory(vk_device, image, image_memory, 0) != VK_SUCCESS)
        return fail("image memory");

    gpu::Texture* texture = gpu::create_external_texture(device, {
        .type = gpu::TextureType::two_d_array,
        .extent = {.x = size, .y = size, .z = 1},
        .layer_count = layers,
        .format = gpu::Format::rgba8_srgb,
        .mutable_format = true,
        .usage = gpu::TextureUsage::color_attachment | gpu::TextureUsage::transfer_source,
    }, image);
    gpu::RenderView* view = texture ? gpu::create_render_view(texture, {.layer_count = layers, .format = gpu::Format::rgba8_unorm}) : nullptr;
    gpu::TimelineSemaphore* timeline = gpu::create_timeline_semaphore(device);
    gpu::CommandPool* pool = gpu::create_command_pool(device);
    if (!texture || !view || !timeline || !pool)
        return fail("external texture, its UNORM view, timeline or pool");

    // The application's frame: acquire, clear both layers through the UNORM view (multiview), release
    gpu::CommandBuffer* commands = gpu::begin_commands(pool);
    gpu::begin_external_access(commands, texture);
    const gpu::ColorAttachment colors[] = {{.render_view = view, .load = gpu::LoadOp::clear, .clear = {0.6f, 0.2f, 1.0f, 1.0f}}};
    gpu::begin_render_pass(commands, {.colors = colors, .view_mask = (1u << layers) - 1u});
    gpu::end_render_pass(commands);
    gpu::end_external_access(commands, texture);
    gpu::end_commands(commands);
    gpu::submit(device, {.commands = {commands}, .completion = {.semaphore = timeline, .value = 1}});
    gpu::wait_timeline({.semaphore = timeline, .value = 1});

    // The "runtime": the image must be in COLOR_ATTACHMENT_OPTIMAL; it copies it to a host-visible buffer on the same queue
    const VkDeviceSize bytes = VkDeviceSize(size) * size * layers * 4;
    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
    if (vkCreateBuffer(vk_device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
        return fail("vkCreateBuffer");
    vkGetBufferMemoryRequirements(vk_device, buffer, &requirements);
    if (!find_memory(vk_physical, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory_type))
        return fail("no host-visible memory");
    const VkMemoryAllocateInfo buffer_allocation{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memory_type,
    };
    void* mapped = nullptr;
    if (vkAllocateMemory(vk_device, &buffer_allocation, nullptr, &buffer_memory) != VK_SUCCESS ||
        vkBindBufferMemory(vk_device, buffer, buffer_memory, 0) != VK_SUCCESS ||
        vkMapMemory(vk_device, buffer_memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
        return fail("buffer memory");
    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = handles.queue_family_index,
    };
    VkCommandPool runtime_pool = VK_NULL_HANDLE;
    VkCommandBuffer runtime_commands = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(vk_device, handles.queue_family_index, handles.queue_index, &queue);
    const VkCommandBufferAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkCreateCommandPool(vk_device, &pool_info, nullptr, &runtime_pool) != VK_SUCCESS)
        return fail("vkCreateCommandPool");
    VkCommandBufferAllocateInfo allocate = allocate_info;
    allocate.commandPool = runtime_pool;
    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkAllocateCommandBuffers(vk_device, &allocate, &runtime_commands) != VK_SUCCESS ||
        vkBeginCommandBuffer(runtime_commands, &begin_info) != VK_SUCCESS)
        return fail("runtime command buffer");
    const VkImageMemoryBarrier2 to_copy{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = layers},
    };
    const VkDependencyInfo to_copy_dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_copy,
    };
    vkCmdPipelineBarrier2(runtime_commands, &to_copy_dependency);
    const VkBufferImageCopy region{
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = layers},
        .imageExtent = {size, size, 1},
    };
    vkCmdCopyImageToBuffer(runtime_commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    const VkMemoryBarrier2 to_host{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
    };
    const VkDependencyInfo to_host_dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &to_host,
    };
    vkCmdPipelineBarrier2(runtime_commands, &to_host_dependency);
    if (vkEndCommandBuffer(runtime_commands) != VK_SUCCESS)
        return fail("vkEndCommandBuffer");
    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &runtime_commands,
    };
    if (vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(queue) != VK_SUCCESS)
        return fail("runtime submit");

    // 0.6 through the UNORM view: 153 (an sRGB view would have encoded it to 203); 0.2 -> 51; 1.0 -> 255
    int result = 0;
    const uint8* pixels = static_cast<const uint8*>(mapped);
    for (uint32 index = 0; index < size * size * layers; ++index)
    {
        const uint8* pixel = pixels + index * 4;
        if (pixel[0] != 153 || pixel[1] != 51 || pixel[2] != 255 || pixel[3] != 255)
        {
            fprintf(stderr, "external texture: pixel %u = (%u, %u, %u, %u), expected (153, 51, 255, 255)\n", index, pixel[0], pixel[1],
                    pixel[2], pixel[3]);
            result = 1;
            break;
        }
    }

    vkDestroyCommandPool(vk_device, runtime_pool, nullptr);
    vkDestroyBuffer(vk_device, buffer, nullptr);
    vkFreeMemory(vk_device, buffer_memory, nullptr);
    gpu::destroy_command_pool(pool);
    gpu::destroy_timeline_semaphore(timeline);
    gpu::destroy_render_view(view);
    gpu::destroy_texture(texture); // leaves the image: it is the "runtime's"
    vkDestroyImage(vk_device, image, nullptr);
    vkFreeMemory(vk_device, image_memory, nullptr);
    gpu::destroy_device(device);
    if (result == 0)
        printf("external texture: hooks, UNORM view of an sRGB image, layouts: OK\n");
    return result;
}
