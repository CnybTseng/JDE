import email
import smtplib
import argparse
import os.path as osp
from mimetypes import guess_type
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.header import Header

def parse_args():
    parser = argparse.ArgumentParser(
        description='Send email through Python.')
    parser.add_argument('--host', type=str,
        default='smtp.qq.com',
        help='E-mail host. Default: smtp.qq.com')
    parser.add_argument('--sender', '-s', type=str, nargs='+',
        help='Sender name and E-mail address.'
            ' E.G., `Zhangwei xxxxxx@163.com`.')
    parser.add_argument('--license', '-l', type=str,
        help='E-mail license from E-mail server.')
    parser.add_argument('--receiver', '-r', type=str, nargs='+',
        help='Receiver name and E-mail address.'
            ' E.G., `Xiaoming xxxxxx@qq.com`.')
    parser.add_argument('--subject', '-sub', type=str,
        default='AI Server',
        help='E-mail subject')
    parser.add_argument('--text', '-t', type=str,
        default='All work have been done. Check the result please!',
        help='E-mail content')
    parser.add_argument('--image', '-i', type=str, nargs='+',
        help='Image list to be attached.')
    parser.add_argument('--excel', '-e', type=str, nargs='+',
        help='Excels list to be attached')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    mail_host = "smtp.qq.com"
    mail_sender = args.sender[1]
    mail_license = args.license
    mail_receivers = [args.receiver[1]]

    mm = MIMEMultipart('related')

    mm['From'] = "{}<{}>".format(args.sender[0], args.sender[1])
    mm['To'] = "{}<{}>".format(args.receiver[0], args.receiver[1])
    mm['subject'] = Header(args.subject, 'utf-8')

    # Attach text
    mesg = MIMEText(args.text, 'plain', 'utf-8')
    mm.attach(mesg)

    # Attach images.
    if args.image is not None:
        for file in args.image:
            with open(file, 'rb') as im:
                mimetype, encoding = guess_type(file)
                maintype, subtype = mimetype.split('/');
                mimeim = MIMEImage(im.read(), **{'_subtype': subtype})
            _, name_with_ext = osp.split(file)
            mimeim['Content-Disposition'] = \
                'attachment;filename={}'.format(name_with_ext)
            mm.attach(mimeim)
    
    # Attach excels.
    if args.excel is not None:
        for file in args.excel:
            with open(file, 'rb') as txt:
                mimetxt = MIMEText(txt.read(), 'base64', 'utf-8')
            _, name_with_ext = osp.split(file)
            mimetxt['Content-Disposition'] = \
                'attachment;filename={}'.format(name_with_ext)
            mm.attach(mimetxt)

    stp = smtplib.SMTP()
    stp.connect(mail_host, 25)
    stp.set_debuglevel(1)
    stp.login(mail_sender, mail_license)
    stp.sendmail(mail_sender, mail_receivers, mm.as_string())
    stp.quit()