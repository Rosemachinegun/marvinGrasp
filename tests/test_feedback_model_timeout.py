from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from grasp_core.communication import request_ik_feedback as feedback


def reader_stub():
    reader = feedback.MeasuredToolPoseReader.__new__(feedback.MeasuredToolPoseReader)
    reader.client = Mock()
    reader.executor = Mock()
    reader.params = None
    reader.fk_by_hand = {}
    reader.service = Mock()
    reader.GetParameters = SimpleNamespace(Request=SimpleNamespace)
    return reader


def test_parameter_response_timeout_is_specific_and_cancels_request(monkeypatch):
    reader = reader_stub()
    future = Future()
    reader.service.call_async.return_value = future
    monkeypatch.setattr(feedback.time, 'monotonic', lambda: 20.)
    with pytest.raises(RuntimeError, match='get_parameters did not respond'):
        reader._load_model('left', 10., lambda: False)
    assert future.cancelled()
    assert reader.params is None


def test_service_discovery_timeout_is_specific(monkeypatch):
    reader = reader_stub()
    reader.service.service_is_ready.return_value = False
    monkeypatch.setattr(feedback.time, 'monotonic', lambda: 20.)
    with pytest.raises(RuntimeError, match='get_parameters service unavailable'):
        reader._load_model('left', 10., lambda: False)
    reader.service.call_async.assert_not_called()


def test_model_loading_does_not_consume_feedback_timeout(monkeypatch):
    reader = reader_stub()
    now = [100.]
    monkeypatch.setattr(feedback.time, 'monotonic', lambda: now[0])
    def load(hand, deadline, cancelled):
        assert deadline == 115.
        now[0] = 112.
    reader._load_model = load
    reader.node = Mock()
    reader.node.get_clock.return_value.now.return_value.nanoseconds = 0
    def spin(deadline, cancelled):
        assert deadline == 117.
        raise RuntimeError('end of test')
    reader._spin = spin
    with pytest.raises(RuntimeError, match='end of test'):
        reader.read('left', timeout_sec=5., cancelled=lambda: False, require_settled=True)
