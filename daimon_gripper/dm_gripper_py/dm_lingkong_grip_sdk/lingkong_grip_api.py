from .grpc_client import CanClient
import time
import threading
from ctypes import c_uint64


class LingkongGrip():
    
    def __init__(self,
                 server_address: str = "0.0.0.0:55551", 
                 interface: str = "can0", 
                 bitrate: int = 1000000,
                 loopback: bool = False,
                 recv_own_msgs: bool = False,
                 connect_attempts: int = 3,
                 connect_timeout_sec: float = 5.0,
                 connect_retry_delay_sec: float = 0.5):
        self.client = CanClient(
            server_address,
            connect_attempts=connect_attempts,
            connect_timeout_sec=connect_timeout_sec,
            connect_retry_delay_sec=connect_retry_delay_sec,
        )
        self.init_status = self.client.init_can(interface, bitrate, loopback, recv_own_msgs)
        self._clamp_pos = None          
        self._open_pos = None           
        self._cur_pos = None
        self._can_id = 0x141
        self._speed = 50 
        self._cur_encode = None
        self._pos_lock = threading.Lock()
        self._error_status = 0
        self._latest_news_time = time.time() * 1000
        self._torque_limit = 0
        self._cur_temprature = 0
        self._latest_pos = None
        self._enable_print = False
        self._cur_current = 0
        self._speed_coe = 1000
        self._init_flag = False
        self._request_running = False
        self._request_thread = None
        self._closed = False


    def grip_init(self, time_out=3000):
        """Electric claw initialization"""
        if not self.init_status:
            print("Remote CAN communication failed to initialize")
            return False
        print("Waiting for initialization to complete")
        self.client.recv_can_async(self._on_message_received, 1000)
        time.sleep(0.1)
        # First send the speed command to clamp the electric gripper
        clamp_cmd = [0x00]*8
        clamp_cmd[0] = 0xA2
        
        temp_data = self._speed * self._speed_coe
        for i in range(4):
            clamp_cmd[i+4] = temp_data % 256
            temp_data //= 256
        cur_time = time.time() * 1000
        self.client.send_can(self._can_id, clamp_cmd)
        
        time.sleep(0.2)
        cur_encode = self.read_encode() 
        if cur_encode == -1:
            print("Electric claw clamping response timeout")
            return False
        
        while (time.time()*1000 - cur_time) < time_out:
            self.client.send_can(self._can_id, clamp_cmd)
            time.sleep(0.2)
            if abs(self.read_encode() - cur_encode) > 10:
                cur_encode = self.read_encode()
                print(F"Encoder Position: {cur_encode}")
            else:
                break

        else:
            print("Electric claw clamping timeout")
            return False
        
        self._send_read_pos_cmd()
        time.sleep(0.2)
        if self._cur_pos is None:
            print("Read clamping position timeout")
            return False
        self._clamp_pos = self._cur_pos
        temp_data = -(self._speed * self._speed_coe)
        for i in range(4):
            clamp_cmd[i+4] = temp_data % 256
            temp_data //= 256
        cur_time = time.time() * 1000
        self.client.send_can(self._can_id, clamp_cmd)
        time.sleep(2.0)
        self._send_read_pos_cmd()
        time.sleep(0.2)
        if self._cur_pos == self._clamp_pos:
            print("Read open position")
            return False
        self._open_pos = self._cur_pos
        self._max_itinerary = abs(self._open_pos -self._clamp_pos)
        self._start_request_thread()
        if self._max_itinerary > 40000:
            self._speed_coe = 3600
        self.set_torque_limit(90)
        self.move_to_pos(0)
        time.sleep(1)
        self._send_read_pos_cmd()
        time.sleep(0.2)
        self._send_read_pos_cmd()
        time.sleep(0.2)
        self._clamp_pos = self._cur_pos
        if self._speed_coe == 3600:
            self._max_itinerary = 90000 
        else:
            self._max_itinerary = 25000
        self._open_pos = self._clamp_pos - self._max_itinerary
        print(F"clamp_pos: {self._clamp_pos}")
        print(F"open_pos: {self._open_pos}")
        print(F"max_itinerary: {self._max_itinerary}")
        self._init_flag = True
        self.move_to_pos(100)
        return True

    def grip_init_with_known_limits(
        self,
        clamp_pos: int,
        open_pos: int,
        max_itinerary: int,
        speed_coe: int = 3600,
        status_timeout: float = 2.0,
    ):
        """Initialize from known calibration values without running homing."""
        if not self.init_status:
            print("Remote CAN communication failed to initialize")
            return False
        if max_itinerary <= 0:
            print(F"Invalid max_itinerary: {max_itinerary}")
            return False

        self._clamp_pos = int(clamp_pos)
        self._open_pos = int(open_pos)
        self._max_itinerary = int(max_itinerary)
        self._speed_coe = int(speed_coe)

        self.client.recv_can_async(self._on_message_received, 1000)
        self._start_request_thread()
        self._init_flag = True

        deadline = time.time() + max(float(status_timeout), 0.1)
        while time.time() < deadline:
            self._send_read_pos_cmd()
            self._send_read_status_cmd()
            time.sleep(0.05)
            if self._cur_pos is not None:
                print(F"clamp_pos: {self._clamp_pos}")
                print(F"open_pos: {self._open_pos}")
                print(F"max_itinerary: {self._max_itinerary}")
                print(F"speed_coe: {self._speed_coe}")
                return True

        print("Read current position timeout after known calibration init")
        return False
        
    def set_speed(self, speed:int):
        """Set the gripper speed, range is 10~100
        """
        if (speed) == self._speed:
            return True
        if speed < 10 or speed > 100:
            print(F"Input position: {speed} is not within the range of 10~100")
            return False
        self._speed = speed
        if self._latest_pos is not None:
            self.move_to_pos(self._latest_pos)
        return True
        
    def read_encode(self):
        """Read the value of the encoder
        """
        if self._cur_encode is not None:
            return self._cur_encode
        else:
            return -1       


    def _to_uint(self, value, bits=16):
        mask = (1 << bits) - 1
        return value & mask
    
    def move_to_pos(self, pos:int):
        """Move to the specified location"""
        if pos < 0 or pos > 1000:
            print(F"Input position: {pos} is not within the range 0~1000")
            return False
        
        cmd_data = [0x00]*8
        cmd_data[0] = 0xA4
        # Speed setting value * coefficient
        cmd_speed = self._speed * self._speed_coe
        cmd_data[2] = cmd_speed // 100 % 256
        cmd_data[3] = cmd_speed // 100 // 256
        # Target position calculation
        target_pos = self._open_pos + int((1000 - pos) /1000 * self._max_itinerary)
        for i in range(4):
            cmd_data[i+4] = target_pos % 256
            target_pos //= 256
        self.client.send_can(self._can_id, cmd_data)
        self._latest_pos = pos
        return True
        
    def read_pos(self):
        """Read the current position, the normal range is 0~1000
        """
        if self._error_status != 0:
            print(F"Motor status abnormal, error code: {self._error_status}")
            return -1
        if (time.time() * 1000 - self._latest_news_time) > 2000:
            print("Remote communication interrupted, please check the communication status")
            return -1
        if self._cur_pos is not None and self._clamp_pos is not None:
            pos = 1000 - int((self._cur_pos - self._open_pos) / self._max_itinerary * 1000)
            return pos
            
        else:
            return -1
        
    def set_torque_limit(self, torque_limit:int):
        if torque_limit == self._torque_limit:
            return True
        if torque_limit < 10 or torque_limit > 100:
            print(F"Input position: {torque_limit} is not within the range of 10~100")
            return False
        torque_limit = (torque_limit * 5)
        read_pos_cmd_data = [0x00]*8
        read_pos_cmd_data[0] = 0xC1
        read_pos_cmd_data[1] = 30
        read_pos_cmd_data[4] = torque_limit % 256
        read_pos_cmd_data[5] = torque_limit //256
        self.client.send_can(self._can_id, read_pos_cmd_data)  
        return True       
    
    def read_torque_limit(self):
        return self._torque_limit
    
    def read_cur_tempture(self):
        return self._cur_temprature

    def read_cur_current(self):
        return self._cur_current
    
    def read_speed(self):
        return self._speed

    def read_error_status(self):
        """Return the latest motor error bit mask."""
        return self._error_status

    def clear_motor_error(self, timeout: float = 1.0):
        """Clear latched motor errors and confirm that status returns to normal."""
        clear_cmd = [0x00] * 8
        clear_cmd[0] = 0x9B
        if not self.client.send_can(self._can_id, clear_cmd):
            return False

        deadline = time.time() + max(float(timeout), 0.1)
        while time.time() < deadline:
            self._send_read_status_cmd()
            time.sleep(0.05)
            if self._error_status == 0:
                return True
        return False
    
    def enable_print(self, enalbe:bool):
        self._enable_print = enalbe
    
    def close(self, reset_torque: bool = True):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._request_running = False
        client = getattr(self, "client", None)
        if getattr(self, "_init_flag", False) and reset_torque and client is not None:
            self.set_torque_limit(50)
            time.sleep(0.1)
        thread = getattr(self, "_request_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        if client is not None:
            client.close()
        
    def _update_torque_limit(self):
        read_pos_cmd_data = [0x00]*8
        read_pos_cmd_data[0] = 0xC0
        read_pos_cmd_data[1] = 30
        self.client.send_can(self._can_id, read_pos_cmd_data)   
                
    def _send_read_pos_cmd(self):
        read_pos_cmd_data = [0x00]*8
        read_pos_cmd_data[0] = 0x92
        self.client.send_can(self._can_id, read_pos_cmd_data)

    def _send_read_status_cmd(self):
        read_pos_cmd_data = [0x00]*8
        read_pos_cmd_data[0] = 0x9A
        self.client.send_can(self._can_id, read_pos_cmd_data)

    def _start_request_thread(self):
        thread = getattr(self, "_request_thread", None)
        if thread is not None and thread.is_alive():
            return
        self._request_running = True
        self._request_thread = threading.Thread(target=self._send_request, daemon=True)
        self._request_thread.start()

    def _send_request(self):
        delt_time=0.02
        num = 0
        while self._request_running:
            self._send_read_pos_cmd()
            time.sleep(delt_time)
            num += 1
            if num % 10 == 0:
                self._send_read_status_cmd()
                num = 0

    def _on_message_received(self, msg):
        """Message receiving callback function
        """
        if msg['can_id'] != self._can_id:
            print(F"Received message CAN ID is not equal to {self._can_id}")
            return
        self._latest_news_time = time.time() * 1000 
        recv_data = list(msg['data'])
        if len(recv_data)!=8:
            print("The length of the returned data is not equal to 8")
            return
        if recv_data[0]==0x92:
            self._cur_pos = self._seven_uint8_to_int64(recv_data[1:])
            
        elif recv_data[0]==0xA2:
            self._cur_encode = recv_data[7] * 256 + recv_data[6]
        elif recv_data[0]==0x9A:
            self._error_status = recv_data[7]
            self._cur_temprature = recv_data[1]
            self._cur_current = recv_data[5] * 256 + recv_data[4]

        elif recv_data[0]==0xC0 or recv_data[0]==0xC1:
            self._torque_limit = int((recv_data[5] * 256 + recv_data[4]) / 5)
   
    def __del__(self):
        self.close()
        
    def _seven_uint8_to_int64(self, data: list) -> int:
        if len(data) != 7:
            raise ValueError(f"7 uint8 values are required, but {len(data)} were provided")
        
        for i, value in enumerate(data):
            if value < 0 or value > 255:
                raise ValueError(f"The {i} value is out of the uint8 range: {value}")
        
        result = 0
        for i, byte in enumerate(data):
            result |= (byte << (i * 8))
        
        if result & (1 << 55):
            result -= (1 << 56)
        
        return result


        
if __name__ == "__main__":
    grip = LingkongGrip(server_address="192.168.127.10:55551")
    grip.grip_init()
    
    grip.set_torque_limit(50)
    
    grip.move_to_pos(1000)
    # i = 0
    # time.sleep(2)

    # while True:

    #     time.sleep(1)
    #     i+=1
    #     print(F"Number of readings: {i}, Current position {grip.read_pos()}, Current temperature: {grip.read_cur_tempture()}, Current torque: {grip.read_torque_limit()}")
    #     if (grip.read_pos() == -1):
    #         break
    #     grip.move_to_pos(0)

        
