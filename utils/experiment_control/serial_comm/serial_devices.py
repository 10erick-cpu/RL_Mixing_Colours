import time

import serial


class ConnectionException(ValueError):
    def __init__(self, e):
        super(ConnectionException, self).__init__(e)


class SerialDevice(object):
    def __init__(self, port='/dev/ttyUSB1', properties=None):
        self.connection = None
        self.port = port
        self.conn_properties = properties
        self.debug = False

    def _create_device(self):
        dev = serial.Serial()
        dev.port = self.port
        dev.baudrate = 9600
        dev.parity = serial.PARITY_NONE
        dev.stopbits = serial.STOPBITS_ONE
        dev.bytesize = serial.EIGHTBITS
        # dev.parity = serial.PARITY_ODD
        # dev.stopbits = serial.STOPBITS_TWO
        # dev.bytesize = serial.SEVENBITS
        dev.timeout = 1.5

        return dev

    def connect(self):
        #print("connecting")
        if self.connection is None:
            self.connection = self._create_device()
        if self.connection.is_open:
            return True
        else:
            try:
                self.connection.open()

                # self.logger.debug("connected to %s" % self.port)
            except serial.SerialException as s_ex:
                raise ConnectionException(s_ex)
            if not self.connected():
                raise ConnectionException("Unable to connect to device")
        return True

    def disconnect(self):
        print("disconnecting")
        if not self.connected():
            return True
        try:
            self.connection.close()
        except serial.SerialException as s_ex:
            raise ConnectionException(s_ex)
        # self.logger.debug("disconnected from %s" % self.port)
        return True

    def connected(self):
        return self.connection is not None and self.connection.is_open

    def send_data(self, data, receive_data=False, receive_eol="\n"):

        if not self.connected():
            self.connect()
        self.connection.flushInput()
        data_bytes = data.encode()

        # self.logger.debug("Writing %s" % data_bytes)
        start = time.time()
        print("tx", data_bytes)
        self.connection.write(data_bytes)
        delta = time.time() - start
        # self.logger.debug("writing to socket took {}s".format(delta))
        if not receive_data:
            # self.disconnect()
            return None

        return self.recv_data(eol_char=receive_eol)

    def recv_data(self, eol_char="\n", auto_decode=True, disconnect=False):
        if not self.connected():
            self.connect()

        data = self.connection.read_until(eol_char.encode("UTF-8"))
        print("rcv", data)
        if auto_decode:
            data = data.decode("utf-8")
        data = data if len(data) > 0 else None
        if disconnect:
            self.disconnect()
        return data

    @staticmethod
    def for_local_port(port):
        return SerialDevice(port)


class VirtualSerialDevice(SerialDevice):
    class VirtualRawDevice(object):
        def __init__(self, port, receiver):
            self.port = port
            self.is_open = False
            self.last_msg = None
            self.receiver = receiver
            if self.receiver is None:
                raise ValueError("Receiver is none")

        def open(self):
            self.is_open = True
            if self.receiver:
                self.receiver.on_open(self.port)

        def close(self):
            self.is_open = False
            if self.receiver:
                self.receiver.on_close(self.port)

        def write(self, bits):
            self.last_msg = bits.decode("utf-8")
            if self.receiver:
                self.receiver.on_write(self.port, bits)

        def read_until(self, eol):
            return "F".encode("UTF-8")

        def flushInput(self):
            pass

        def readline(self):
            return str.encode("DummyLine | lastmsg: " + self.last_msg)

        def set_receiver(self, receiver):
            self.receiver = receiver

    def __init__(self, port, properties=None, receiver=None):
        super().__init__(port, properties)
        self.receiver = receiver

    def _create_device(self):
        return VirtualSerialDevice.VirtualRawDevice(self.port, self.receiver)
