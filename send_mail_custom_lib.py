import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time

class EmailSender:
    def __init__(self, smtp_server, smtp_port, email_from, email_pass):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_from = email_from
        self.email_pass = email_pass
    
    def send_email(self, email_to, subject, body, attachment_filename=None):
        # Ensure email_to is a list
        if isinstance(email_to, str):
            email_to = [email_to]

        msg = MIMEMultipart()
        msg['From'] = self.email_from
        msg['To'] = ', '.join(email_to)
        msg['Subject'] = subject

        # Attach the body of the message
        if body:
            msg.attach(MIMEText(body, 'plain'))

        # Attach the file if provided
        if attachment_filename:
            try:
                with open(attachment_filename, 'rb') as attachment:
                    attachment_package = MIMEBase('application', 'octet-stream')
                    attachment_package.set_payload(attachment.read())
                    encoders.encode_base64(attachment_package)
                    attachment_package.add_header('Content-Disposition', f'attachment; filename= {attachment_filename}')
                    msg.attach(attachment_package)
            except FileNotFoundError:
                print(f"Attachment file '{attachment_filename}' not found.")
                return

        text = msg.as_string()

        try:
            print("Sending Mail ...... ")
            server = smtplib.SMTP(host=self.smtp_server, port=self.smtp_port)
            server.starttls()
            server.login(user=self.email_from, password=self.email_pass)

            print(f"Sending Email to: {', '.join(email_to)}")
            server.sendmail(from_addr=self.email_from, to_addrs=email_to, msg=text)
            print("Email Successfully Sent")

        except Exception as e:
            print(f"Failed to send email: {e}")

        finally:
            server.quit()

# Example usage
if __name__ == "__main__":
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    email_from = "meharchetandeepsingh@gmail.com"
    email_pass = ""

    email_to = ["iotnc4@gmail.com"]  # Can be a string or a list
    subject = "New email from Python video attachment"
    body = "Hello"

    attachment_filename = 'example.mp4'  # Change to None if no attachment

    email_sender = EmailSender(smtp_server, smtp_port, email_from, email_pass)

    start = time.time()
    email_sender.send_email(email_to, subject, body, attachment_filename)
    end = time.time()

    elapsed_time = end - start
    print(f"Elapsed Time: {elapsed_time} seconds")
