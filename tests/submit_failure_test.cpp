// A failed vkQueueSubmit2 (other than a lost device) is fatal in every build: the error callback gets a fatal report naming it before
// the abort, so the application never waits for a timeline value the failed submission would have signaled.
#include <NoGraphicsAPI/NoGraphicsAPI.hpp>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace
{
bool fail_submit = false;

VkResult VKAPI_CALL test_vkQueueSubmit2(VkQueue queue, uint32 count, const VkSubmitInfo2* submits, VkFence fence)
{
    return fail_submit ? VK_ERROR_OUT_OF_DEVICE_MEMORY : vkQueueSubmit2(queue, count, submits, fence);
}
}

// Inject only this translation unit's native submission
#define vkQueueSubmit2 test_vkQueueSubmit2
#include "../src/NoGraphicsAPI.cpp"

namespace
{
void receive_error(const char* message, bool fatal) noexcept
{
    if (!fail_submit) return;
    if (fatal && strstr(message, "vkQueueSubmit2") && strstr(message, "Vulkan result -2"))
    {
        printf("A failed submission is reported as fatal: %s\n", message);
        fflush(stdout);
        _Exit(0);
    }
    fprintf(stderr, "FAIL: unexpected report for the failed submission: %s (fatal %d)\n", message, int(fatal));
    _Exit(1);
}
}

int main()
{
    gpu::set_error_callback(receive_error);
    const gpu::DeviceInit init = gpu::create_device();
    if (init.error == gpu::Error::unsupported)
    {
        printf("SKIP: required Vulkan device features are unavailable.\n");
        return 77;
    }
    if (!init.device) return 1;
    gpu::CommandPool* pool = gpu::create_command_pool(init.device);
    gpu::TimelineSemaphore* timeline = gpu::create_timeline_semaphore(init.device);
    gpu::CommandBuffer* commands = gpu::begin_commands(pool);
    gpu::end_commands(commands);
    fail_submit = true;
    gpu::submit(init.device, {.commands = {commands}, .completion = {.semaphore = timeline, .value = 1}}, 0);
    fprintf(stderr, "FAIL: a failed vkQueueSubmit2 returned to the application without a fatal report\n");
    return 1;
}
