import ctypes

import warp as wp
import warp._src.context as wp_context


class _CpuStream:
    """Stand-in for wp.Stream on CPU, where launches execute synchronously."""

    def wait_stream(self, other_stream):
        pass


def resolve_launch_device(device, stream, block_dim, use_cuda_graph):
    device = wp.get_device(device) if stream is None else stream.device
    if device.is_cpu:
        if use_cuda_graph:
            raise ValueError("use_cuda_graph=True requires a CUDA device")
        # tile kernels on CPU always run with a single thread per block
        block_dim = 1
    return device, block_dim


def default_stream(device):
    return device.stream if device.is_cuda else _CpuStream()


def create_stream(device):
    return wp.Stream(device) if device.is_cuda else _CpuStream()


def module_name(name, *params):
    # Kernel variants need distinct module names, otherwise loading a second
    # variant on CPU fails to resolve its kernel symbols.
    suffix = "_".join(getattr(p, "__name__", str(p)) for p in params)
    return f"{name}_{suffix}"


def create_cuda_graph_callback(callback, device=None, stream=None):
    with wp.ScopedCapture(device=device, stream=stream) as capture:
        callback()

    graph = capture.graph

    if stream is not None:
        if stream.device != graph.device:
            raise RuntimeError(f"Cannot launch graph from device {graph.device} on stream from device {stream.device}")

    if graph.device.is_cuda and graph.graph_exec is None:
        launch_stream = stream if stream is not None else graph.device.stream
        graph_exec = ctypes.c_void_p()
        result = wp_context.runtime.core.wp_cuda_graph_create_exec(
            graph.device.context,
            launch_stream.cuda_stream,
            graph.graph,
            ctypes.byref(graph_exec),
        )
        if not result:
            raise RuntimeError(f"Graph creation error: {wp_context.runtime.get_error_string()}")
        graph.graph_exec = graph_exec

    def graph_callback():
        wp.capture_launch(graph, stream=stream)

    return graph_callback
