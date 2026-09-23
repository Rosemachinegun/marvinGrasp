import grpc
import logging
from typing import Optional, Generator
import threading
import time

from . import can_service_pb2
from . import can_service_pb2_grpc

class CanClient:
    """CAN gRPC Client
    """
    def __init__(
        self,
        server_address: str = "localhost:50051",
        *,
        connect_attempts: int = 3,
        connect_timeout_sec: float = 5.0,
        connect_retry_delay_sec: float = 0.5,
    ):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        self.connect(
            attempts=connect_attempts,
            timeout_sec=connect_timeout_sec,
            retry_delay_sec=connect_retry_delay_sec,
        )
        self._send_lock = threading.Lock()
        self._send_data_num = 0
        self._read_data_num = 0
    
    def connect(self, attempts: int = 3, timeout_sec: float = 5.0, retry_delay_sec: float = 0.5):
        """Connect to gRPC server
        """
        attempts = max(int(attempts), 1)
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                self.channel = grpc.insecure_channel(
                    self.server_address,
                    options=(("grpc.enable_http_proxy", 0),),
                )
                self.stub = can_service_pb2_grpc.CanServiceStub(self.channel)

                grpc.channel_ready_future(self.channel).result(timeout=timeout_sec)
                self.is_connected = True
                self.logger.info(f"Successfully connected to the server: {self.server_address}")
                return True

            except Exception as e:
                detail = str(e) or type(e).__name__
                last_error = detail
                self.is_connected = False
                if self.channel is not None:
                    self.channel.close()
                    self.channel = None
                    self.stub = None
                if attempt < attempts:
                    self.logger.error(
                        f"Failed to connect to the server {self.server_address} "
                        f"(attempt {attempt}/{attempts}): {detail}; retrying"
                    )
                    time.sleep(max(float(retry_delay_sec), 0.0))
                else:
                    self.logger.error(
                        f"Failed to connect to the server {self.server_address} "
                        f"(attempt {attempt}/{attempts}): {detail}"
                    )
        self.logger.error(
            f"Unable to connect to the server {self.server_address}: {last_error}"
        )
        return False
    
    def init_can(self, interface: str = "can0", 
                 bitrate: int = 500000,
                 loopback: bool = False,
                 recv_own_msgs: bool = False) -> bool:
        """Initialize CAN interface
        """
        try:
            if not self.is_connected:
                if not self.connect():
                    return False
            
            request = can_service_pb2.CanInitRequest(
                interface=interface,
                bitrate=bitrate,
                loopback=loopback,
                recv_own_msgs=recv_own_msgs
            )
            
            response = self.stub.InitCan(request)
            
            if response.success:
                self.logger.info(f"{interface} initialization successful")
            else:
                self.logger.error(f"CAN interface initialization failed: {response.message}")
            
            return response.success
            
        except grpc.RpcError as e:
            self.logger.error(f"RPC call failed: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            self.logger.error(f"Error initializing CAN interface: {e}")
            return False
    
    def send_can(self, can_id: int, data: list,
                 is_extended: bool = False,
                 is_rtr: bool = False) -> bool:
        """Send CAN message
        """
        try:
            if not self.is_connected:
                self.logger.error("Not connected to the server")
                return False
            data = bytes(data)
            request = can_service_pb2.CanMessage(
                can_id=can_id,
                data=data,
                is_extended=is_extended,
                is_rtr=is_rtr
            )
            self._send_lock.acquire()
            response = self.stub.SendCan(request)
            self._send_data_num += 1
            self._send_lock.release()
            
            if response.success:
                pass
            else:
                self.logger.error(f"CAN message sending failed: {response.message}")
            
            return response.success
            
        except grpc.RpcError as e:
            self.logger.error(f"RPC call failed: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            self.logger.error(f"Error sending CAN message: {e}")
            return False
    
    def recv_can(self, timeout_ms: Optional[int] = None,
                 can_id_filter: Optional[int] = None) -> Generator:
        """Receive CAN Message (Generator)
        """
        try:
            if not self.is_connected:
                self.logger.error("Not connected to the server")
                return
            
            request = can_service_pb2.CanRecvRequest()
            
            if timeout_ms is not None:
                request.timeout_ms = timeout_ms
            if can_id_filter is not None:
                request.can_id_filter = can_id_filter
            
            responses = self.stub.RecvCan(request)
            
            for response in responses:
                self._read_data_num += 1
                yield {
                    'can_id': response.can_id,
                    'data': response.data,
                    'is_extended': response.is_extended,
                    'is_rtr': response.is_rtr,
                    'timestamp': response.timestamp
                }
            
        except grpc.RpcError as e:
            if e.code() != grpc.StatusCode.CANCELLED:
                self.logger.error(f"RPC call failed: {e}")
                self.is_connected = False
        except Exception as e:
            self.logger.error(f"Error receiving CAN message: {e}")
    
    def recv_can_async(self, callback, timeout_ms: Optional[int] = None,
                       can_id_filter: Optional[int] = None):
        """Asynchronously receive CAN messages
        """
        def recv_thread():
            try:
                for msg in self.recv_can(timeout_ms, can_id_filter):
                    if callback:
                        callback(msg)
            except Exception as e:
                self.logger.error(f"Asynchronous receiving thread error: {e}")
        
        thread = threading.Thread(target=recv_thread, daemon=True)
        thread.start()
        return thread
    
    def close(self):
        """Close connection
        """
        if self.is_connected:
            self.channel.close()
            self.is_connected = False
        self.logger.info("The connection has been closed")
            

    def print_statistics_num(self):
        print(f"send_num:{self._send_data_num}, recv_num:{self._read_data_num}")
    
    
