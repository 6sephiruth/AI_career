import os
import smtplib

from email.mime.text import MIMEText



def exists(pathname):
    return os.path.exists(pathname)

def mkdir(dir_names):
    for d in dir_names:
        if not os.path.exists(d):
            os.mkdir(d)

def email_send():
    # 아직 코드 미 작성
    # 보안 코드 작성 뭐시기 해야함 ㅜ
    # https://yeolco.tistory.com/93
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('6sephiruth@gmail.com')