import GPUtil
import smtplib, ssl
import time
from datetime import datetime

if __name__ == "__main__":
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "lilymos0427@gmail.com"  # Enter your address
    receiver_email = "mimic6636@gmail.com"  # Enter receiver address
    password = "fjtkuxfsxuatimah"
    message = """\
    Subject: Hi there

    GPU is available!!"""
    sent_email_count = 0
    while True:
        print(GPUtil.showUtilization(all=False, attrList=None, useOldCode=False))
        # deviceIDs = GPUtil.getAvailable(order='first', limit=2, maxLoad=1, maxMemory=1)
        deviceIDs = GPUtil.getAvailable(order='first', limit=2, maxLoad=1, maxMemory=0.4)
        current_time = datetime.now().strftime("%H:%M:%S")
        # print(deviceIDs)
        print("Current Time =", current_time)

        # if 1 in deviceIDs:
        if (len(deviceIDs) > 0):
            sent_email_count += 1
            for gpu in deviceIDs:
                message = message + "\nGPU: " + str(gpu)
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, message)
        
        else:
            sent_email_count = 0
        time.sleep(60 + sent_email_count * 60)
                