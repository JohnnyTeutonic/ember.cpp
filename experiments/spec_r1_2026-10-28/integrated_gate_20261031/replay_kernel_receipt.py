"""Replay the public lossless kernel records only, without CUDA or full model rows."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import check


def replay(receipt):
    if receipt['schema']!='ember.raw-kernel-records.hex.v1':raise ValueError('wrong schema')
    if receipt['source_packet_commit']!='153f0e9352e14cfbd2b6c7328cb32e8d0f6a7ce5':raise ValueError('wrong source')
    if receipt['canonical_zip_sha256']!='d7fa39d85bc8392a4197e90cba1a575fbaec8484ae789b579bb0d32ec034140c':raise ValueError('wrong archive')
    records=receipt['fixtures'];inventory=check.kernel_inventory()
    if len(records)!=len(inventory):raise ValueError('incomplete kernel inventory')
    rows=0
    for rec,expected in zip(records,inventory,strict=True):
        v,n,c=expected
        if (rec['vocab'],rec['rows'],rec['case'])!=expected:raise ValueError('changed kernel matrix')
        if hashlib.sha256(check.kernel_input(v,n,c)).hexdigest()!=rec['input_sha256']:raise ValueError('changed kernel input')
        original=bytes.fromhex(rec['original_hex']);partitioned=bytes.fromhex(rec['partition4_hex'])
        if len(original)!=n*20 or original!=partitioned:raise ValueError('kernel field bit mismatch')
        for row in range(n):
            if struct.unpack_from('<ffiii',original,row*20)!=check.oracle(check.kernel_row(c,v,row)):
                raise ValueError('independent semantic oracle mismatch')
        rows+=n
    return {'status':'PASS_PUBLIC_KERNEL_RECORDS_ONLY','kernel_cases':len(records),'kernel_rows':rows,
            'scope':'Lossless CUDA kernel records, independent CPU semantic oracle; not full raw model-row/RNG replay'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args()
    print(json.dumps(replay(json.loads(a.receipt.read_text())),indent=2))
