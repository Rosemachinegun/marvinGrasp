import numpy as np
from unittest.mock import Mock, patch

from grasp_core.apps.flowpose_request_ik_app import GraspDemoApp, KEY_VOICE, RuntimeState
from grasp_core.ui.tablet_ui import (
    TabletCommand,
    TabletTaskLoopBridge,
    TabletWebService,
)


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
    bridge.request(TabletCommand.VOICE)
    bridge.request(TabletCommand.STOP)

    assert bridge.drain_commands() == [
        TabletCommand.PERCEIVE,
        TabletCommand.GRASP,
        TabletCommand.VOICE,
        TabletCommand.STOP,
    ]
    assert bridge.drain_commands() == []


def test_tablet_voice_uses_existing_v_key_path_on_main_loop():
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.tablet_bridge = TabletTaskLoopBridge()
    app.state = RuntimeState()
    app.manual_home_future = None
    app.handle_key = Mock(return_value=True)
    app.tablet_bridge.request(TabletCommand.VOICE)

    app.handle_tablet_commands(bundle=None)

    app.handle_key.assert_called_once_with(KEY_VOICE, None)


def test_tablet_voice_is_ignored_while_paused():
    app = GraspDemoApp.__new__(GraspDemoApp)
    app.tablet_bridge = TabletTaskLoopBridge()
    app.state = RuntimeState(paused=True)
    app.manual_home_future = None
    app.handle_key = Mock(return_value=True)
    app.tablet_bridge.request(TabletCommand.VOICE)

    app.handle_tablet_commands(bundle=None)

    app.handle_key.assert_not_called()
    assert app.state.status == "Command ignored: robot is stopped / paused"


def test_tablet_service_passes_voice_switch_to_ui():
    bridge = TabletTaskLoopBridge()
    service = TabletWebService(
        bridge, host="127.0.0.1", port=7860, voice_enabled=True,
    )

    with patch("grasp_core.ui.tablet_ui.create_tablet_interface") as build_ui:
        build_ui.return_value.queue.return_value = Mock()
        service.start()

    build_ui.assert_called_once_with(bridge, voice_enabled=True)
