#!/usr/bin/env python3
"""Safely extract the larger labeling archive into a new, separate directory."""
import json
import shutil
import stat
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def main():
    archive=Path('Labeling_Data (1).zip');target=Path('Labeling_Data_v2').resolve()
    if target.exists():raise FileExistsError(target)
    with ZipFile(archive) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        total=sum(i.file_size for i in files)
        if shutil.disk_usage(target.parent).free < total+20*1024**3:
            raise RuntimeError('Insufficient disk: extraction requires 20 GiB checkpoint reserve')
        paths=[]
        for info in files:
            p=PurePosixPath(info.filename)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='Labeling_Data'
            assert not stat.S_ISLNK(info.external_attr>>16)
            destination=target.joinpath(*p.parts[1:]);assert destination.is_relative_to(target)
            paths.append(destination)
        assert len(paths)==len(set(paths))
        target.mkdir();done=0;start=time.monotonic()
        for i,(info,destination) in enumerate(zip(files,paths),1):
            destination.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
            assert destination.stat().st_size==info.file_size
            done+=info.file_size
            if i%200==0 or i==len(files):
                print(json.dumps(dict(files=i,total_files=len(files),GiB=done/1024**3,
                    total_GiB=total/1024**3,elapsed_seconds=time.monotonic()-start)),flush=True)
        (target/'extraction_complete.json').write_text(json.dumps(dict(archive=str(archive.resolve()),
            files=len(files),bytes=total,zip_crc_verified=True),indent=2))


if __name__=='__main__':main()
