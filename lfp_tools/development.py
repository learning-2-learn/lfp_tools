from lfp_tools import general
from lfp_tools import analysis
import configparser
import os.path as op
import s3fs

def get_fs():
    CP = configparser.ConfigParser()
    CP.read_file(open(op.join(op.expanduser('~'), '.aws', 'credentials')))
    CP.sections()
    ak = CP.get('default', 'AWS_ACCESS_KEY_ID')
    sk = CP.get('default', 'AWS_SECRET_ACCESS_KEY')
    fs = s3fs.S3FileSystem(key=ak, secret=sk)
    return(fs)



