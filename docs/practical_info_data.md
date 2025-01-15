# Some Practical Information on Data and Training

## Data Transfer

### Copy Data from Surfdrive / Research Drive to Remote Machine (Google Colab/Snellius)

You can copy your data to [Surfdrive](https://surfdrive.surf.nl), [UvA Research Drive](https://uva.data.surfsara.nl), or [SURF Research Drive](https://researchdrive.surfsara.nl) by zipping your data and using drag and drop to transfer the data. Then, use the share icon to share your data with a password and set it to have no expiration time. Use the "Copy to Clipboard" option to obtain the link. You will receive a link similar to `https://surfdrive.surf.nl/files/index.php/s/IS00bWerWu3MDJS`. The last element, e.g., `IS00bWerWu3MDJS`, represents your username. Utilize this username along with the password you specified when sharing this folder, as shown in the signature below, to download your data.

```bash
curl -u username:password surf_public_link -o your_output_file
```

**Public link**

- The Surfdrive: https://surfdrive.surf.nl/files/public.php/webdav
- The UvA Research Drive: https://uva.data.surfsara.nl/public.php/webdav
- The SURF Research Drive: https://researchdrive.surfsara.nl/public.php/webdav

For the example provided above, here is the code to download the data using curl and then unzip the data. The entire process of obtaining the data and unzipping it took less than 2 minutes for 2.2GB of data. When using with the Google Colab, remember to prefix each command with the ! sign.

```bash
curl -u "IS00bWerWu3MDJS:password" "https://surfdrive.surf.nl/files/public.php/webdav" -o mot
unzip mot -d mot_data > /dev/null 2>&1
```

There is some information on [SURF wiki for Research Drive](https://servicedesk.surf.nl/wiki/display/WIKI/Uploading+files+to+a+Public+link), [SURF wiki for SURFdrive](https://servicedesk.surf.nl/wiki/display/WIKI/Activating+WebDAV) or older one on [surfnet for SURFdrive](https://wiki.surfnet.nl/display/SURFdrive/Accessing+files+via+WebDAV), but I found it unclear.

### Copy Data from the Google Drive to the Google Colab

The data can be dragged and dropped. Alternatively, you can copy your data using other methods:

The below option is for sharing single file. It can be a zip file.

Share the file with "Share/Share/Anyone with the link". Then "Share/Copy Link". You get the url like this:
`https://drive.google.com/file/d/id_to_copy/view?usp=drive_link`. Use `id_to_copy` in `gdown`:

```python
! pip install -U gdown requests
! gdown id_to_copy --output /content/
```

The other option is to mount the whole Google drive (not recommanded):

```python
from google.colab import drive
drive.mount("/content/drive")
!cp /content/test.yaml "/content/drive/MyDrive"
```

### Copy Data from Crunchomics to Snellius and Reverse

N.B. You can only copy data via Crunchomics machine. Port 22 is closed on Snellius.

```bash
# copy from Crunchomics to Snellius
scp -r test.txt username@snellius.surf.nl:/home/username/test.txt
rsync -avz --progress test.txt username@snellius.surf.nl:/home/username/test.txt

# copy from Snellius to Crunchomics
scp -r username@snellius.surf.nl:/home/username/test.txt .
rsync -avz --progress username@snellius.surf.nl:/home/username/test.txt .
```

### Copy Data from Snellius to SRC (Surf Research Cloud)

In Snellius, generate ssh key and add it to SRC. Then create the environment in SRC. In Snellius, you can for example ssh, rsync and scp to this machine. If you have created a Jupyter notebook, the ip address can be found in the "Workspace" under "Details" of this Jupyter notebook. 

This option doesn't work for Crunchomics. Example:

```bash 
ssh fkariminej@145.38.194.242
scp fkariminej@145.38.194.242:/home/fkariminej/test.txt .
```

### Copy Data from Local Machine to SURF Filesender

You can upload data from your local machine and send it to a recipient using [filesender.py](https://github.com/filesender/filesender/blob/master/scripts/client/filesender.py). To view the required arguments, run:

```bash
python filesender.py --help
```

The required arguments are:

```bash
  -u USERNAME, --username USERNAME       # Your username
  -a APIKEY, --apikey APIKEY             # Your API key
  -r RECIPIENTS, --recipients RECIPIENTS # Recipient's email address
  -b BASE_URL, --base_url BASE_URL       # Base URL of the SURF Filesender
  -f FROM_ADDRESS, --from_address FROM_ADDRESS # Your email address
```

To obtain the **API key** and **username**, follow these steps:
1. Go to "My Profile" on [SURF Filesender](https://filesender.surf.nl).
2. Copy the **"Secret"** value for the API key to use with `-a,--apikey`.
3. Copy the **"Identifiant"** value for the username to use with `-u,--username`.

The **base URL** (`-b, --base_url`) is always `https://filesender.surf.nl/rest.php`.

The **from address** (`-f, --from_address`) is your email address, and the **recipients** (`-r, --recipients`) is the recipient's email address. At the end of the command, specify the file you want to send. 

Here is an example command:

```bash
python filesender.py -b https://filesender.surf.nl/rest.php -u my_username_key -a my_api_key -r recipient_email@uva.nl -f my_email@gmail.com my_file_to_be_sent.zip
```

**Prerequisites**: Before using this script, make sure you have the following Python packages installed:

```bash
pip install requests urllib3 cryptography
```

## References

- [Surfdrive](https://surfdrive.surf.nl)
- [SRC(SURF Research Cloud)](https://sram.surf.nl)
- [SURF Filesender](https://filesender.surf.nl)
