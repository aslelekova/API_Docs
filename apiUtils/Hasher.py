import passlib.context
from passlib.hash import hex_sha512 as hex_sha512


class HasherClass:
    def __init__(self):
        self.ImageHasher = passlib.context.CryptContext(schemes=['sha512_crypt'], deprecated='auto')

    def CreateImageFileNameHash(self, FileName: str) -> str:
        return hex_sha512.hash(FileName) + "." + FileName.split(".")[-1]  # type: ignore
