import ctypes

import warp as wp
import warp._src.context as wp_context


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
