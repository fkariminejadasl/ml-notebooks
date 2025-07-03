# Set Up a Custom Python Function within PostgreSQL

Before proceeding with the following steps, ensure you have set up a new environment on your system with a custom Python virtual environment located at `/scratch/venv/envs/p310`. For additional setup instructions, refer to [here](./gradio_src.md#install-miniconda).

Follow these steps to get everything working from scratch:

---

### Step 1: Install PostgreSQL and `plpython3u`

#### Install PostgreSQL and the PL/Python Extension (`plpython3u`):

Run the following commands to install PostgreSQL and the `plpython3u` extension:

```bash
sudo apt update
sudo apt install postgresql-plpython3-14
```

#### Enable `plpython3u` in Your Database:

To enable `plpython3u` in a specific database (e.g., `test`), first log into PostgreSQL:

```bash
sudo -u postgres psql
```

Then, create your test database (if it doesn’t already exist) and enable the extension:

```bash
CREATE DATABASE test; 
\connect test
CREATE EXTENSION plpython3u;
```

Exit the `psql` interface:

```bash
\q
```

---

### Step 2: Create a PostgreSQL User for `fkariminej`

To facilitate authentication, create a PostgreSQL user matching your Linux system user.

#### Create a PostgreSQL User:

Open the PostgreSQL shell as the `postgres` superuser:

```bash
sudo -u postgres psql
```

Create the PostgreSQL user `fkariminej` with superuser privileges:

```bash
CREATE USER fkariminej WITH SUPERUSER CREATEDB CREATEROLE;
```

Exit the shell:

```bash
\q
```

#### Access the Database:

From now on, you can directly access the database with the following command:

```bash
psql -U fkariminej -d test
```

#### Ensure Access Permissions:

Make sure the `postgres` user has access to your home directory and the relevant paths:

```bash
sudo chmod o+rx /home/fkariminej
sudo chmod o+rx /scratch
sudo chmod o+rx /scratch/venv
sudo chmod o+rx /scratch/venv/envs
sudo chmod o+rx /scratch/venv/envs/p310
```

---

### Step 3: Create a Shell Script to Start PostgreSQL with Python Environment Variables

#### Create a Shell Script to Start PostgreSQL:

Open a new shell script file for PostgreSQL:

```bash
vi /scratch/pg_service.sh
```

#### Add Environment Variables for Python Paths and Executable:

Add the following lines to set the `PYTHONPATH` and `PYTHONUSERBASE` for your virtual environment:

```bash
#!/bin/bash

export PGDATABASE=test
export PGUSER=fkariminej
export PGPORT=5432
export PATH=/usr/lib/postgresql/14/bin:$PATH
export PYTHONUSERBASE=/scratch/venv/envs/p310
export PYTHONPATH=/scratch/venv/envs/p310/lib/python310.zip:/scratch/venv/envs/p310/lib/python3.10:/scratch/venv/envs/p310/lib/python3.10/lib-dynload:/scratch/venv/envs/p310/lib/python3.10/site-packages

# Start PostgreSQL using the correct configuration file
pg_ctl -D /etc/postgresql/14/main -l /var/lib/postgresql/14/main/postgres_logfile start
```

#### Make the Script Executable:

```bash
chmod +x /scratch/pg_service.sh
```

#### Create and Set Permissions for the Logfile:

```bash
sudo touch /var/lib/postgresql/14/main/postgres_logfile
sudo chown postgres:postgres /var/lib/postgresql/14/main/postgres_logfile
```

#### Stop the Currently Running PostgreSQL Instance:

```bash
sudo systemctl stop postgresql@14-main.service
```

You can find the services:

```bash
systemctl list-units --type=service | grep postgresql
```

#### Run the Script as the `postgres` User:

```bash
sudo -u postgres /scratch/pg_service.sh
```

#### Verify the Configuration:

After running the script, verify the setup:

```bash
psql -U fkariminej -d test
```

#### Create and Test a PL/Python Function:

Create a PL/Python function and run it to ensure everything works:

```bash
CREATE OR REPLACE FUNCTION pymtorch()
RETURNS float8 AS $$
import sys
import torch
x = torch.Tensor([1, 3]) * torch.Tensor([4, 3]).to(torch.float32)
return float(x.sum())
$$ LANGUAGE plpython3u;

SELECT pymtorch();
```

---

### Step 4: Create a New Systemd Service for Automation

To avoid manually stopping and restarting PostgreSQL every time the machine starts, create a new service:

#### Create a New Systemd Service File:

```bash
sudo vi /etc/systemd/system/newpostgres.service
```

#### Add the Following Configuration:

```bash
[Unit]
Description=Custom PostgreSQL Service using /scratch/pg_service.sh
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c "sudo systemctl stop postgresql@14-main.service && sudo -u postgres /scratch/pg_service.sh"
ExecStop=/bin/bash -c "sudo -u postgres /usr/lib/postgresql/14/bin/pg_ctl -D /etc/postgresql/14/main stop"
User=fkariminej

[Install]
WantedBy=multi-user.target
```

#### Reload Systemd and Enable the New Service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable newpostgres.service
sudo systemctl start newpostgres.service
sudo systemctl status newpostgres.service
```

#### Clean Up Residual Files if Needed:

If issues arise, clean up any leftover files:

```bash
sudo rm -f /var/lib/postgresql/14/main/postmaster.pid
sudo rm -f /var/run/postgresql/.s.PGSQL.5432
```

#### Check for Active Connections:

```bash
sudo lsof -i :5432
```

#### Verify PostgreSQL Processes Are Not Running:

```bash
ps aux | grep postgres
sudo kill -9 <PID>
```

#### Re-enable the New Service:

```bash
sudo systemctl enable newpostgres.service
```

---

### Notes

> The `postgresql@14-main.service` is located in `/lib/systemd/system/{postgresql@.service, postgresql.service}`. Their symbolic links are found here:

```bash
'/etc/systemd/system/multi-user.target.wants/postgresql@14-main.service' -> '/lib/systemd/system/postgresql@.service'
/etc/systemd/system/multi-user.target.wants/postgresql.service -> /lib/systemd/system/postgresql.service
```

#### Other important paths:

```bash
/usr/lib/postgresql/14/bin
/var/lib/postgresql/14/main  # owner:group -> postgres:postgres
/etc/postgresql/14/main      # pg_ctl.conf -> owner:group postgres:postgres
```

## SQL Commands Cheat Sheet

```bash
/*
multiple comments
*/
CREATE DATABASE test; 
DROP DATABASE test; 
\connect test 
CREATE EXTENSION plpython3u; 
CREATE USER fkariminej WITH SUPERUSER CREATEDB CREATEROLE;
SHOW data_directory; -- PGDATA SHOW data_directory;
SELECT current_user; -- PGUSER here fkariminej
SELECT current_database(); -- PGDATABASE here test
\df  -- Show list of functions

-- A specific function (e.g. my_func), a specific schema (e.g., my_schema)
SELECT my_func(); -- Run the function
DROP FUNCTION my_func; -- Remove a function
DROP FUNCTION my_func(integer, text); -- Remove a function with parameter
DROP FUNCTION my_schema.my_func; -- Remove a function exists in a specific schema
DROP FUNCTION IF EXISTS my_func;

-- Print the definition of a function
SELECT pg_get_functiondef(oid) 
FROM pg_proc 
WHERE proname = 'my_func';

-- Print the definition of a function with Schema Context
SELECT pg_get_functiondef(p.oid) 
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE p.proname = 'my_func' AND n.nspname = 'my_schema';

-- Output the function definition in a file
\o output_file.sql
SELECT pg_get_functiondef(oid) 
FROM pg_proc 
WHERE proname = 'my_func';
\o

\q -- Exit
```

```bash
sudo -u postgres psql
sudo -u postgres /scratch/pg_service.sh
psql -U fkariminej -d test
```