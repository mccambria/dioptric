# -*- coding: utf-8 -*-
"""
Created on August 19th, 2021

Email notifications test

@author: mccambria
"""

# Email imports
import smtplib
from email.mime.text import MIMEText
import socket
import time
import traceback
import keyring


def send_exception_email():
    # format_exc extracts the stack and error message from
    # the exception currently being handled.
    now = time.localtime()
    date = time.strftime(
        "%A, %B %d, %Y",
        now,
    )
    timex = time.strftime("%I:%M:%S %p", now)
    exc_info = traceback.format_exc()
    content = (
        f"An unhandled exception occurred on {date} at {timex}.\n{exc_info}"
    )
    send_email(content)


def send_email(
    content,
    email_from="kolkowitznvlab@gmail.com",
    email_to="kolkowitznvlab@gmail.com",
):

    pc_name = socket.gethostname()
    msg = MIMEText(content)
    msg["Subject"] = f"Alert from {pc_name}"
    msg["From"] = email_from
    msg["To"] = email_to

    pw = keyring.get_password("system", email_from)

    server = smtplib.SMTP("smtp.gmail.com", 587)  # port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(email_from, pw)
    server.sendmail(email_from, email_to, msg.as_string())
    server.close()


if __name__ == "__main__":
    try:
        test = 1 / 0
    except Exception as exc:
        # Intercept the exception so we can email it out and re-raise it
        send_exception_email()
        raise exc
    finally:
        print("here")
