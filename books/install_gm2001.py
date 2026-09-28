"""Download and verify the optional local GM2001 Polyglot opening book."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://github.com/ChrisWhittington/polyglot-books/releases/download/v1.0/gm2001.zip"
ARCHIVE_SHA256 = "68514dcedac0394dbda5b4cae1a834f274e1b1e7fde5dce310c107924c03aa6a"
BOOK_SHA256 = "fb6e9f3f27bb19a5b2fdefcc441c88ddeae48db61d5f00ad83973abb9f939c87"
TARGET = Path(__file__).with_name("gm2001.bin")


def main():
    if TARGET.is_file() and sha256(TARGET.read_bytes()).hexdigest() == BOOK_SHA256:
        print(f"Opening book already verified: {TARGET}")
        return
    with urlopen(URL, timeout=60) as response:
        archive = response.read()
    if sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("GM2001 archive checksum mismatch")
    with ZipFile(BytesIO(archive)) as bundle:
        book = bundle.read("gm2001.bin")
    if sha256(book).hexdigest() != BOOK_SHA256:
        raise RuntimeError("GM2001 book checksum mismatch")
    temporary = TARGET.with_suffix(".bin.tmp")
    temporary.write_bytes(book)
    temporary.replace(TARGET)
    print(f"Installed verified opening book: {TARGET}")


if __name__ == "__main__":
    main()
