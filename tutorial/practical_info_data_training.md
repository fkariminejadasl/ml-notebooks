# Some Practical Information on Data and Training

## Data Transfer

### Copy Data from Surfdrive to Remote Machine (Google Colab/Snellius)

You can copy your data to Surfdrive by zipping your data and using drag and drop to transfer the data. Then, use the share icon to share your data with a password and set it to have no expiration time. Use the "Copy to Clipboard" option to obtain the link. You will receive a link similar to `https://surfdrive.surf.nl/files/index.php/s/IS00bWerWu3MDJS`. The last element, e.g., `IS00bWerWu3MDJS`, represents your username. Utilize this username along with the password you specified when sharing this folder, as shown in the signature below, to download your data.

```bash
curl -u username:password surf_public_link -o your_output_file
```

The Surf public link is: https://surfdrive.surf.nl/files/public.php/webdav

For the example provided above, here is the code to download the data using curl and then unzip the data. The entire process of obtaining the data and unzipping it took less than 2 minutes for 2.2GB of data. When using with the Google Colab, remember to prefix each command with the ! sign.

```bash
curl -u "IS00bWerWu3MDJS:password" "https://surfdrive.surf.nl/files/public.php/webdav" -o mot
unzip mot -d mot_data > /dev/null 2>&1
```

### Copy Data from the Google Drive to the Google Colab
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
