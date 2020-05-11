# libraries to be imported
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import datetime

def mailSending(todaysFile):
    fromaddr = "sunilbhandari1@gmail.com"
    toaddr = "sunilbhandari1@gmail.com,akshat.hi@gmail.com"

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr
    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Latest report as of "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # string to store the body of the mail
    body = "Report on "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # open the file to be sent
    filename = todaysFile
    attachment = open("C:/Users/sunilbhandari1/Desktop/Kiteconnect/"+filename, "rb")

    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    p.set_payload((attachment).read())

    # encode into base64
    encoders.encode_base64(p)

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "sun22Nyu@un")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()
    return