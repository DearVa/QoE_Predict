#!/usr/bin/env python

import getopt
import sys

import dpkt
import pcap


def usage():
    sys.stderr.write('Usage: %s [-i device] [-l] [pattern]' % sys.argv[0])
    sys.stderr.write("""
Options:

\t-i device - Use the specific device.
\t-l - Use pcap.loop() method.

Available devices:""")
    sys.stderr.write('\t' + '\n\t'.join(pcap.findalldevs()))
    sys.exit(1)


def iter(pc, decode_fn):
    for ts, pkt in pc:
        msg = '%d %r' % (ts, decode_fn(pkt))
        print(msg)


def loop(pc, decode_fn):
    def cb(ts, pkt, *args):
        msg = '%d %r' % (ts, decode_fn(pkt))
        print(msg)
    pc.loop(0, cb)


def main():
    opts, args = getopt.getopt(sys.argv[1:], 'i:hl')
    name = None
    use_loop = False
    for o, a in opts:
        if o == '-i':
            name = a
        elif o == '-l':
            use_loop = True
        else:
            usage()

    pc = pcap.pcap(name, timeout_ms=50)
    pc.setfilter(' '.join(args))
    decode = {
        pcap.DLT_LOOP: dpkt.loopback.Loopback,
        pcap.DLT_NULL: dpkt.loopback.Loopback,
        pcap.DLT_EN10MB: dpkt.ethernet.Ethernet
    }[pc.datalink()]

    print('listening on %s: %s' % (pc.name, pc.filter))
    try:
        if use_loop:
            loop(pc, decode)
        else:
            iter(pc, decode)
    except KeyboardInterrupt:
        nrecv, ndrop, nifdrop = pc.stats()
        print('\n%d packets received by filter' % nrecv)
        print('%d packets dropped by kernel' % ndrop)

if __name__ == '__main__':
    main()
