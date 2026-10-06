"""Virtual ethernet hub for testing the cluster in QEMU on Windows.

Each QEMU node sends its ethernet frames as UDP datagrams to 127.0.0.1:4200;
the hub copies every frame to all other nodes (ports 4201..4204), exactly like
a dumb ethernet hub. Not part of the OS - only for the emulated test cluster.
"""
import socket

HUB = ("127.0.0.1", 4200)
NODES = [("127.0.0.1", 4200 + n) for n in range(1, 5)]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(HUB)
print(f"hub listening on {HUB[0]}:{HUB[1]}, nodes on ports 4201-4204")
while True:
    try:
        frame, sender = sock.recvfrom(65536)
    except ConnectionResetError:   # Windows: a node port is not open (yet)
        continue
    for node in NODES:
        if node != sender:
            try:
                sock.sendto(frame, node)
            except OSError:
                pass
