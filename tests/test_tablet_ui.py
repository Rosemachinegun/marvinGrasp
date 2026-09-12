import numpy as np

from grasp_core.ui.tablet_ui import TabletCommand, TabletTaskLoopBridge


def test_bridge_keeps_raw_frame_and_status_snapshot_isolated():
    bridge = TabletTaskLoopBridge()
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[0, 0] = [10, 20, 30]
    sam_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    sam_bgr[0, 1] = [1, 2, 3]
    flowpose_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    flowpose_bgr[1, 0] = [4, 5, 6]

    bridge.publish(bgr, "Ready", sam_bgr=sam_bgr, flowpose_bgr=flowpose_bgr)
    snapshot = bridge.snapshot()

    assert snapshot.status == "Ready"
    assert snapshot.image_rgb[0, 0].tolist() == [30, 20, 10]
    assert snapshot.sam_rgb[0, 1].tolist() == [3, 2, 1]
    assert snapshot.flowpose_rgb[1, 0].tolist() == [6, 5, 4]
    snapshot.image_rgb[:] = 255
    snapshot.sam_rgb[:] = 255
    snapshot.flowpose_rgb[:] = 255
    assert bridge.snapshot().image_rgb[0, 0].tolist() == [30, 20, 10]
    assert bridge.snapshot().sam_rgb[0, 1].tolist() == [3, 2, 1]
    assert bridge.snapshot().flowpose_rgb[1, 0].tolist() == [6, 5, 4]


def test_bridge_queues_commands_in_order():
    bridge = TabletTaskLoopBridge()
    bridge.request(TabletCommand.PERCEIVE)
    bridge.request(TabletCommand.GRASP)
    bridge.request(TabletCommand.STOP)

    assert bridge.drain_commands() == [
        TabletCommand.PERCEIVE,
        TabletCommand.GRASP,
        TabletCommand.STOP,
    ]
    assert bridge.drain_commands() == []
