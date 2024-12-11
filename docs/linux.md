# Linux

## Shared Directories

Linux distributions typically follow the Filesystem Hierarchy Standard (FHS), which defines where different types of files are placed. Some common system-wide (shared) directories are:

- **/etc**: System-wide configuration files.  
- **/usr**: Contains the majority of userland programs and data.  
  - **/usr/bin**: Executables for most user programs.  
  - **/usr/sbin**: Executables for system administration.  
  - **/usr/lib**: Shared libraries and internal binaries used by programs in `/usr/bin` and `/usr/sbin`.  
  - **/usr/local**: For programs installed locally (outside of the distribution’s package manager), typically in subdirectories like `/usr/local/bin` and `/usr/local/lib`.  
  - **/usr/share**: Architecture-independent data such as icons, documentation, and locale files.
- **/var**: Variable data that changes during system operation.  
  - **/var/lib**: State information and variable data for system programs.
  - **/var/log**: Log files.
- **/opt**: Optional or third-party software, often self-contained packages.
- **/lib**, **/lib64**: Core system libraries (for `/bin` and `/sbin`).

When you install packages using a package manager like `apt`, the files typically go into these shared directories depending on their type:

- Executables usually end up in `/usr/bin` or `/usr/sbin`.
- Libraries go into `/usr/lib`.
- Configuration files go into `/etc`.
- Documentation and data files often end up in `/usr/share`.

These directories are accessible system-wide, so all users on the system can use the installed programs without additional per-user setup.


## Cheat Sheet

#### File and Directory Management

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `ls`                             | List files and directories in the current directory.                        |
| `ls -l`                          | List files with detailed information.                                       |
| `ls -a`                          | List all files, including hidden ones.                                      |
| `cd [directory]`                 | Change the current directory to the specified one.                          |
| `pwd`                            | Display the current working directory path.                                 |
| `mkdir [directory]`              | Create a new directory.                                                     |
| `rmdir [directory]`              | Remove an empty directory.                                                  |
| `rm [file]`                      | Delete a file.                                                              |
| `rm -r [directory]`              | Recursively delete a directory and its contents.                            |
| `cp [source] [destination]`      | Copy files or directories.                                                  |
| `mv [source] [destination]`      | Move or rename files or directories.                                        |
| `touch [file]`                   | Create an empty file or update the timestamp of an existing file.           |
| `cat [file]`                     | Display the contents of a file.                                             |
| `less [file]`                    | View the contents of a file one screen at a time.                           |
| `head [file]`                    | Display the first 10 lines of a file.                                       |
| `tail [file]`                    | Display the last 10 lines of a file.                                        |
| `diff [file1] [file2]`           | Compare two files line by line and display the differences.                 |
| `readlink [link]`                | Display the target of a symbolic link.                                      |
| `readlink -f [path]`             | Display the absolute path, resolving all symbolic links.                    |
| `ln [target] [link_name]`        | Create a hard link to a file.                                               |
| `ln -s [target] [link_name]`     | Create a symbolic (soft) link to a file or directory.                       |

#### File Permissions and Ownership

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `chmod [permissions] [file]`     | Change the permissions of a file or directory.                              |
| `chown [owner]:[group] [file]`   | Change the owner and group of a file or directory.                          |
| `chgrp [group] [file]`           | Change the group of a file or directory.                                    |

#### Process Management

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `ps`                             | Display information about active processes.                                 |
| `pgrep: ps + grep`               | Search for processes by name or other attributes and display their PIDs.    |
| `top`                            | Display real-time system information, including active processes.           |
| `htop`                           | Interactive process viewer (requires installation).                         |
| `nvtop`                          | Interactive monitor for NVIDIA GPUs (requires NVIDIA GPUs).                 |
| `kill [PID]`                     | Terminate a process by its Process ID (PID).                                |
| `killall [process_name]`         | Terminate all processes with the specified name.                            |
| `bg`                             | Resume a suspended job in the background.                                   |
| `fg`                             | Bring a background job to the foreground.                                   |

#### Disk Usage and Storage

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `df -h`                          | Display disk space usage in human-readable format.                          |
| `du -sh [directory]`             | Display the size of a directory and its contents.                           |
| `mount [device] [mount_point]`   | Mount a device to the filesystem.                                           |
| `umount [device]`                | Unmount a device from the filesystem.                                       |

#### Networking

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `ifconfig`                       | Display or configure network interfaces.                                    |
| `ip a`                           | Display all network interfaces and their IP addresses.                      |
| `ping [host]`                    | Send ICMP ECHO_REQUEST packets to network hosts.                            |
| `wget [url]`                     | Download files from the internet.                                           |
| `curl [url]`                     | Transfer data from or to a server.                                          |
| `ssh [user]@[host]`              | Connect to a remote host via SSH.                                           |

#### File Transfer

| Command                                     | Description                                                                 |
|---------------------------------------------|-----------------------------------------------------------------------------|
| `scp [source] [user@host:destination]`      | Securely copy files between hosts over a network.                           |
| `rsync [options] [source] [destination]`    | Synchronize files and directories between two locations efficiently.        |

#### User Management

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `adduser [username]`             | Create a new user.                                                          |
| `passwd [username]`              | Change the password for a user.                                             |
| `deluser [username]`             | Delete a user.                                                              |
| `usermod -aG [group] [username]` | Add a user to a group.                                                      |

#### System Information

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `uname -a`                       | Display all system information.                                             |
| `uname -r`                       | Display the kernel version.                                                 |
| `uptime`                         | Show how long the system has been running.                                  |
| `date`                           | Display or set the system date and time.                                    |
| `who`                            | Show who is logged into the system.                                         |
| `whoami`                         | Display the current logged-in user's username.                              |

#### Package Management (Debian-based systems)

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `apt update`                     | Update the package index.                                                   |
| `apt upgrade`                    | Upgrade all installed packages to their latest versions.                    |
| `apt install [package]`          | Install a new package.                                                      |
| `apt remove [package]`           | Remove an installed package.                                                |
| `apt search [package]`           | Search for a package in the repositories.                                   |

#### Text Processing

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `grep [pattern] [file]`          | Search for a pattern in a file.                                             |
| `sed 's/[old]/[new]/' [file]`    | Replace text in a file using stream editor.                                 |
| `awk '{print $1}' [file]`        | Pattern scanning and processing language.                                   |

#### Compression and Archiving

| Command                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `tar -cvf [archive.tar] [files]` | Create a tarball archive of files.                                          |
| `tar -xvf [archive.tar]`         | Extract files from a tarball archive.                                       |
| `gzip [file]`                    | Compress a file using gzip.                                                 |
| `gunzip [file.gz]`               | Decompress a gzip compressed file.                                          |


This cheat sheet provides a quick reference to common Linux commands. For more detailed information, refer to the manual pages by typing `command --help` or `man command` in the terminal. 


## Examples

```bash
# Copy a local file to a remote host:
scp /path/to/local/file.txt user@remote_host:/path/to/remote/directory/

# Copy a file from a remote host to the local machine:
scp user@remote_host:/path/to/remote/file.txt /path/to/local/directory/

# Synchronize a local directory to a remote host:
# `-a`: Archive mode (preserves permissions, times, symbolic links, etc.).
# `-v`: Verbose output.
# `-z`: Compress data during transfer.
rsync -avz /path/to/local/directory/ user@remote_host:/path/to/remote/directory/

# Synchronize a remote directory to the local machine:
rsync -avz user@remote_host:/path/to/remote/directory/ /path/to/local/directory/

# Find the process ID(s) of a running program:
pgrep process_name

# Find processes by user:
pgrep -u username

# Find differences
diff --color -U 0 file1 file2

# Display the absolute path
readlink -f $HOME

# To create a symbolic link named `my_link` that points to a file `myfile.txt`
ln -s myfile.txt my_link
```
